export class Callback {
  setup(_trainer, _model, _stage) {}
  teardown(_trainer, _model, _stage) {}
  onFitStart(_trainer, _model) {}
  onFitEnd(_trainer, _model) {}
  onTrainStart(_trainer, _model) {}
  onTrainEnd(_trainer, _model) {}
  onTrainEpochStart(_trainer, _model) {}
  onTrainEpochEnd(_trainer, _model) {}
  onTrainBatchStart(_trainer, _model, _batch, _batchIdx) {}
  onTrainBatchEnd(_trainer, _model, _outputs, _batch, _batchIdx) {}
  onValidationStart(_trainer, _model) {}
  onValidationEnd(_trainer, _model) {}
  onValidationEpochStart(_trainer, _model) {}
  onValidationEpochEnd(_trainer, _model) {}
  onValidationBatchStart(_trainer, _model, _batch, _batchIdx) {}
  onValidationBatchEnd(_trainer, _model, _outputs, _batch, _batchIdx) {}
  onTestStart(_trainer, _model) {}
  onTestEnd(_trainer, _model) {}
  onTestBatchStart(_trainer, _model, _batch, _batchIdx) {}
  onTestBatchEnd(_trainer, _model, _outputs, _batch, _batchIdx) {}
  onPredictStart(_trainer, _model) {}
  onPredictEnd(_trainer, _model) {}
  onPredictBatchStart(_trainer, _model, _batch, _batchIdx) {}
  onPredictBatchEnd(_trainer, _model, _outputs, _batch, _batchIdx) {}
  onBeforeBackward(_trainer, _model, _loss) {}
  onAfterBackward(_trainer, _model) {}
  onBeforeOptimizerStep(_trainer, _model, _optimizer) {}
  onBeforeZeroGrad(_trainer, _model, _optimizer) {}
  onSaveCheckpoint(_trainer, _model, _checkpoint) {}
  onLoadCheckpoint(_trainer, _model, _checkpoint) {}
}
