export class Callback {
  setup(_trainer: unknown, _model: unknown, _stage: unknown): void {}
  teardown(_trainer: unknown, _model: unknown, _stage: unknown): void {}
  onFitStart(_trainer: unknown, _model: unknown): void {}
  onFitEnd(_trainer: unknown, _model: unknown): void {}
  onTrainStart(_trainer: unknown, _model: unknown): void {}
  onTrainEnd(_trainer: unknown, _model: unknown): void {}
  onTrainEpochStart(_trainer: unknown, _model: unknown): void {}
  onTrainEpochEnd(_trainer: unknown, _model: unknown): void {}
  onTrainBatchStart(_trainer: unknown, _model: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onTrainBatchEnd(_trainer: unknown, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onValidationStart(_trainer: unknown, _model: unknown): void {}
  onValidationEnd(_trainer: unknown, _model: unknown): void {}
  onValidationEpochStart(_trainer: unknown, _model: unknown): void {}
  onValidationEpochEnd(_trainer: unknown, _model: unknown): void {}
  onValidationBatchStart(_trainer: unknown, _model: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onValidationBatchEnd(_trainer: unknown, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onTestStart(_trainer: unknown, _model: unknown): void {}
  onTestEnd(_trainer: unknown, _model: unknown): void {}
  onTestBatchStart(_trainer: unknown, _model: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onTestBatchEnd(_trainer: unknown, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onPredictStart(_trainer: unknown, _model: unknown): void {}
  onPredictEnd(_trainer: unknown, _model: unknown): void {}
  onPredictBatchStart(_trainer: unknown, _model: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onPredictBatchEnd(_trainer: unknown, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {}
  onBeforeBackward(_trainer: unknown, _model: unknown, _loss: unknown): void {}
  onAfterBackward(_trainer: unknown, _model: unknown): void {}
  onBeforeOptimizerStep(_trainer: unknown, _model: unknown, _optimizer: unknown): void {}
  onBeforeZeroGrad(_trainer: unknown, _model: unknown, _optimizer: unknown): void {}
  onSaveCheckpoint(_trainer: unknown, _model: unknown, _checkpoint: unknown): void {}
  onLoadCheckpoint(_trainer: unknown, _model: unknown, _checkpoint: unknown): void {}
}
