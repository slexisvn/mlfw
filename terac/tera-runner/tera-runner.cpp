//===- tera-runner.cpp - JIT driver for the tera dialect --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/JitInvoker.h"
#include "Tera/IR/TeraDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

using namespace mlir;
using namespace mlir::tera;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir file>"),
                                         llvm::cl::init("-"));

llvm::cl::opt<std::string> entryOption(
    "entry", llvm::cl::desc("Name of the function to call"),
    llvm::cl::init("main"));

llvm::cl::opt<std::string> dataSource(
    "data",
    llvm::cl::desc("The entry function's inputs, and optionally its expected "
                   "output: either a path to the JSON or the JSON itself"),
    llvm::cl::init(""));

llvm::cl::opt<bool> check(
    "check",
    llvm::cl::desc("Compare the result against the expected output in --data"),
    llvm::cl::init(false));

llvm::cl::opt<double> tolerance(
    "tolerance", llvm::cl::desc("Relative tolerance used by --check"),
    llvm::cl::init(1e-5));

llvm::cl::opt<unsigned> optLevel("O", llvm::cl::desc("JIT optimisation level"),
                                 llvm::cl::init(3));

llvm::cl::opt<std::string> targetName(
    "target", llvm::cl::desc("Machine to lower the module for"),
    llvm::cl::init("cpu"));

llvm::cl::opt<std::string> targetOptions(
    "target-options",
    llvm::cl::desc("Options for the target's pipeline, as name=value pairs"),
    llvm::cl::init(""));

llvm::cl::opt<unsigned> benchmarkRuns(
    "benchmark",
    llvm::cl::desc("Call the entry this many times and report the timings "
                   "rather than its result"),
    llvm::cl::init(0));

llvm::cl::opt<unsigned> warmupRuns(
    "warmup",
    llvm::cl::desc("Untimed calls before the timed ones. The first call is "
                   "what makes the JIT emit the code, so timing it would "
                   "measure the compiler"),
    llvm::cl::init(3));

llvm::cl::opt<uint64_t> seedOption(
    "seed",
    llvm::cl::desc("Seed for the inputs --benchmark makes up when --data "
                   "supplies none"),
    llvm::cl::init(20260901));

llvm::cl::opt<double> spreadOption(
    "spread",
    llvm::cl::desc("Made-up inputs are drawn uniformly from [-spread, spread]"),
    llvm::cl::init(1.0));

llvm::cl::list<std::string> sharedLibs(
    "shared-libs",
    llvm::cl::desc("Shared libraries to load into the JIT, for the runtime "
                   "helpers a lowered memref program calls"),
    llvm::cl::MiscFlags::CommaSeparated);

LogicalResult readTensor(const llvm::json::Value &value, TensorBuffer &buffer,
                         llvm::StringRef what) {
  const llvm::json::Object *object = value.getAsObject();
  const llvm::json::Array *data = object ? object->getArray("data") : nullptr;
  if (!data) {
    llvm::errs() << "tera-runner: " << what << " is not {\"data\": [...]}\n";
    return failure();
  }
  if (const llvm::json::Array *shape = object->getArray("shape")) {
    SmallVector<int64_t> given;
    for (const llvm::json::Value &extent : *shape)
      given.push_back(extent.getAsInteger().value_or(-1));
    if (ArrayRef<int64_t>(given) != buffer.getType().getShape()) {
      llvm::errs() << "tera-runner: " << what << " has shape " << given.size()
                   << "D, but the function declares " << buffer.getType()
                   << "\n";
      return failure();
    }
  }
  if (static_cast<int64_t>(data->size()) != buffer.getNumElements()) {
    llvm::errs() << "tera-runner: " << what << " holds " << data->size()
                 << " elements, but " << buffer.getType() << " needs "
                 << buffer.getNumElements() << "\n";
    return failure();
  }

  for (auto [index, element] : llvm::enumerate(*data)) {
    std::optional<double> number = element.getAsNumber();
    if (!number) {
      llvm::errs() << "tera-runner: " << what << " holds a non-numeric entry\n";
      return failure();
    }
    buffer.setElement(index, *number);
  }
  return success();
}

llvm::json::Value writeTensor(TensorBuffer &buffer) {
  llvm::json::Array data;
  for (int64_t index = 0; index < buffer.getNumElements(); ++index)
    data.push_back(buffer.getElement(index));

  llvm::json::Array shape;
  for (int64_t extent : buffer.getType().getShape())
    shape.push_back(extent);

  std::string dtype;
  llvm::raw_string_ostream(dtype) << buffer.getType().getElementType();

  return llvm::json::Object{
      {"dtype", dtype}, {"shape", std::move(shape)}, {"data", std::move(data)}};
}

LogicalResult compareTensors(TensorBuffer &actual,
                             const llvm::json::Value &expected,
                             llvm::StringRef what) {
  TensorBuffer reference(actual.getType());
  if (failed(readTensor(expected, reference, what)))
    return failure();

  unsigned reported = 0;
  for (int64_t index = 0; index < actual.getNumElements(); ++index) {
    double got = actual.getElement(index);
    double want = reference.getElement(index);
    if (std::abs(got - want) <= tolerance * std::max(1.0, std::abs(want)))
      continue;
    if (++reported <= 16)
      llvm::errs() << "tera-runner: " << what << " element " << index
                   << " is " << got << ", expected " << want << "\n";
  }
  if (reported == 0)
    return success();
  llvm::errs() << "tera-runner: " << what << ": " << reported << " of "
               << actual.getNumElements() << " elements disagree\n";
  return failure();
}

LogicalResult compareResults(MutableArrayRef<TensorBuffer> results,
                             const llvm::json::Value &expected) {
  const llvm::json::Array *many = expected.getAsArray();
  if (!many) {
    if (results.size() != 1) {
      llvm::errs() << "tera-runner: the entry returns " << results.size()
                   << " tensors, so --data must expect an array of them\n";
      return failure();
    }
    return compareTensors(results.front(), expected, "expected output");
  }
  if (many->size() != results.size()) {
    llvm::errs() << "tera-runner: --data expects " << many->size()
                 << " tensors but the entry returns " << results.size()
                 << "\n";
    return failure();
  }
  for (auto [index, one] : llvm::enumerate(*many)) {
    std::string what = ("expected output " + llvm::Twine(index)).str();
    if (failed(compareTensors(results[index], one, what)))
      return failure();
  }
  return success();
}

FailureOr<llvm::json::Value>
loadData(std::unique_ptr<llvm::MemoryBuffer> &held) {
  if (dataSource.empty())
    return llvm::json::Value(nullptr);

  llvm::StringRef text = dataSource;
  if (!text.starts_with("{")) {
    auto file = llvm::MemoryBuffer::getFile(dataSource);
    if (!file) {
      llvm::errs() << "tera-runner: cannot read " << dataSource << ": "
                   << file.getError().message() << "\n";
      return failure();
    }
    held = std::move(*file);
    text = held->getBuffer();
  }

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    llvm::errs() << "tera-runner: " << toString(parsed.takeError()) << "\n";
    return failure();
  }
  return std::move(*parsed);
}

LogicalResult benchmarkEntry(JitInvoker &invoker, StringRef name,
                             MutableArrayRef<TensorBuffer> inputs,
                             MutableArrayRef<TensorBuffer> results) {
  using Clock = std::chrono::steady_clock;

  for (unsigned iteration = 0; iteration < warmupRuns; ++iteration)
    if (failed(invoker.invoke(name, inputs, results)))
      return failure();

  SmallVector<double> samples;
  samples.reserve(benchmarkRuns);
  for (unsigned iteration = 0; iteration < benchmarkRuns; ++iteration) {
    Clock::time_point started = Clock::now();
    if (failed(invoker.invoke(name, inputs, results)))
      return failure();
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count());
  }

  double total = std::accumulate(samples.begin(), samples.end(), 0.0);
  llvm::sort(samples);
  llvm::outs() << "tera-runner: " << name << ": " << samples.size()
               << " runs, best " << llvm::format("%.4f", samples.front())
               << " ms, median "
               << llvm::format("%.4f", samples[samples.size() / 2])
               << " ms, mean " << llvm::format("%.4f", total / samples.size())
               << " ms, total " << llvm::format("%.4f", total) << " ms\n";
  return success();
}

FailureOr<SmallVector<int64_t>> recordedShape(const llvm::json::Array *given,
                                              size_t index,
                                              RankedTensorType type,
                                              StringRef entryName) {
  const llvm::json::Object *object =
      given && index < given->size() ? (*given)[index].getAsObject() : nullptr;
  const llvm::json::Array *shape =
      object ? object->getArray("shape") : nullptr;
  if (!shape || static_cast<int64_t>(shape->size()) != type.getRank()) {
    llvm::errs() << "tera-runner: " << entryName << " argument " << index
                 << " is " << type
                 << ", so --data has to give its shape\n";
    return failure();
  }
  SmallVector<int64_t> extents;
  for (auto [axis, extent] : llvm::enumerate(*shape)) {
    std::optional<int64_t> value = extent.getAsInteger();
    int64_t declared = type.getDimSize(axis);
    if (!value || *value < 0 ||
        (!ShapedType::isDynamic(declared) && *value != declared)) {
      llvm::errs() << "tera-runner: " << entryName << " argument " << index
                   << " axis " << axis << " disagrees with " << type
                   << "\n";
      return failure();
    }
    extents.push_back(*value);
  }
  return extents;
}

/// The arguments the entry asks to have left on the device, uploaded once and
/// pointed at for every run. What a call is handed for one of these is a device
/// pointer in place of the host one, which is the whole of the calling
/// convention `tera.device_resident` names.
class ResidentInputs {
public:
  ResidentInputs() = default;
  ResidentInputs(const ResidentInputs &) = delete;
  ResidentInputs &operator=(const ResidentInputs &) = delete;

  ~ResidentInputs() {
    for (void *pointer : owned)
      memory->release(pointer);
  }

  LogicalResult place(ArrayRef<unsigned> positions, JitInvoker &invoker,
                      MutableArrayRef<TensorBuffer> inputs) {
    if (positions.empty())
      return success();

    memory = invoker.getDeviceMemory();
    if (!memory) {
      llvm::errs() << "tera-runner: this target has no device memory to leave "
                      "an argument on\n";
      return failure();
    }

    for (unsigned position : positions) {
      TensorBuffer &input = inputs[position];
      size_t bytes = static_cast<size_t>(input.getNumElements()) *
                     input.getElementByteSize();
      void *pointer = memory->allocate(bytes);
      if (!pointer) {
        llvm::errs() << "tera-runner: the device has no room for argument "
                     << position << "\n";
        return failure();
      }
      owned.push_back(pointer);

      MutableArrayRef<uint64_t> descriptor = input.getDescriptor();
      memory->upload(pointer, reinterpret_cast<void *>(descriptor[1]), bytes);
      descriptor[0] = descriptor[1] = reinterpret_cast<uint64_t>(pointer);
    }
    return success();
  }

private:
  DeviceMemory *memory = nullptr;
  SmallVector<void *> owned;
};

SmallVector<unsigned> residentArguments(func::FuncOp entry) {
  SmallVector<unsigned> positions;
  for (unsigned index = 0; index < entry.getNumArguments(); ++index)
    if (entry.getArgAttr(index, TeraDialect::kDeviceResidentAttrName))
      positions.push_back(index);
  return positions;
}

LogicalResult readSignature(TypeRange types, StringRef what,
                            SmallVectorImpl<RankedTensorType> &tensors) {
  for (Type type : types) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType) {
      llvm::errs() << "tera-runner: " << what << " is " << type
                   << ", which the runner cannot materialise\n";
      return failure();
    }
    if (failed(TensorBuffer::checkElementType(tensorType.getElementType(),
                                              "tera-runner: " + what.str())))
      return failure();
    tensors.push_back(tensorType);
  }
  return success();
}

LogicalResult run(ModuleOp module) {
  using Clock = std::chrono::steady_clock;

  std::unique_ptr<llvm::MemoryBuffer> held;
  FailureOr<llvm::json::Value> data = loadData(held);
  if (failed(data))
    return failure();
  const llvm::json::Object *dataObject = data->getAsObject();

  std::string entryName = entryOption;
  if (entryName == "main" && dataObject)
    if (std::optional<llvm::StringRef> named = dataObject->getString("entry"))
      entryName = named->str();

  auto entry = module.lookupSymbol<func::FuncOp>(entryName);
  if (!entry) {
    llvm::errs() << "tera-runner: no function named " << entryName << "\n";
    return failure();
  }
  FunctionType signature = entry.getFunctionType();

  SmallVector<RankedTensorType> inputTypes;
  SmallVector<RankedTensorType> resultTypes;
  if (failed(readSignature(signature.getInputs(), "an argument", inputTypes)))
    return failure();
  if (signature.getNumResults() == 0) {
    llvm::errs() << "tera-runner: " << entryName
                 << " returns nothing\n";
    return failure();
  }
  if (failed(readSignature(signature.getResults(), "the result", resultTypes)))
    return failure();

  const llvm::json::Array *given =
      dataObject ? dataObject->getArray("inputs") : nullptr;
  SmallVector<TensorBuffer> inputs;
  for (auto [index, type] : llvm::enumerate(inputTypes)) {
    if (type.hasStaticShape()) {
      inputs.emplace_back(type);
      continue;
    }
    FailureOr<SmallVector<int64_t>> shape =
        recordedShape(given, index, type, entryName);
    if (failed(shape))
      return failure();
    inputs.emplace_back(type, *shape);
  }
  SmallVector<TensorBuffer> results;
  for (RankedTensorType type : resultTypes)
    results.push_back(TensorBuffer::forResult(type));

  if (dataSource.empty()) {
    std::mt19937_64 generator(seedOption);
    for (TensorBuffer &input : inputs)
      input.fill(generator, spreadOption);
  } else {
    const llvm::json::Array *given =
        dataObject ? dataObject->getArray("inputs") : nullptr;
    size_t provided = given ? given->size() : 0;
    if (provided != inputs.size()) {
      llvm::errs() << "tera-runner: " << entryName << " takes " << inputs.size()
                   << " tensors, but --data supplies " << provided << "\n";
      return failure();
    }
    for (auto [index, input] : llvm::enumerate(inputs))
      if (failed(readTensor((*given)[index], input,
                            "input " + std::to_string(index))))
        return failure();
  }

  const TargetBackend *target = lookupTargetBackend(targetName);
  if (!target) {
    llvm::errs() << "tera-runner: no target named " << targetName << "; this "
                 << "build has";
    for (StringRef name : getTargetBackendNames())
      llvm::errs() << " " << name;
    llvm::errs() << "\n";
    return failure();
  }

  SmallVector<unsigned> resident = residentArguments(entry);

  Clock::time_point started = Clock::now();
  FailureOr<std::unique_ptr<JitInvoker>> invoker =
      JitInvoker::create(module, *target, targetOptions, optLevel, sharedLibs);
  if (failed(invoker))
    return failure();
  double compiled =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();

  ResidentInputs onTheDevice;
  if (failed(onTheDevice.place(resident, **invoker, inputs)))
    return failure();

  if (failed((*invoker)->invoke(entryName, inputs, results)))
    return failure();

  for (TensorBuffer &result : results)
    result.adoptDescriptorShape();

  if (benchmarkRuns) {
    llvm::outs() << "tera-runner: " << entryName << ": compile "
                 << llvm::format("%.1f", compiled) << " ms\n";
  } else if (results.size() == 1) {
    llvm::outs() << llvm::formatv("{0:2}", writeTensor(results.front()))
                 << "\n";
  } else {
    llvm::json::Array all;
    for (TensorBuffer &result : results)
      all.push_back(writeTensor(result));
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(all)))
                 << "\n";
  }

  if (check) {
    const llvm::json::Value *expected =
        dataObject ? dataObject->get("output") : nullptr;
    if (!expected) {
      llvm::errs()
          << "tera-runner: --check needs an output in --data\n";
      return failure();
    }
    if (failed(compareResults(results, *expected)))
      return failure();
  }

  if (!benchmarkRuns)
    return success();
  return benchmarkEntry(**invoker, entryName, inputs, results);
}

}

int main(int argc, char **argv) {
  llvm::InitLLVM initialiser(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  registerPassManagerCLOptions();
  registerTeraTargets();
  llvm::cl::ParseCommandLineOptions(argc, argv, "Tera JIT runner\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<TeraDialect>();
  registerAllGPUToLLVMIRTranslations(registry);

  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return 1;

  return failed(run(*module)) ? 1 : 0;
}
