# Music Segmentation Diagrams

This page is an index. Each diagram lives in its own file so it can be opened
and rendered independently.

## Sequence diagrams

1. [Main segmentation flow](sequence-diagrams/01-main-segmentation-flow.md)
2. [Custom analysis pipeline](sequence-diagrams/02-custom-analysis-pipeline.md)
3. [Evaluation compare flow](sequence-diagrams/03-evaluation-compare-flow.md)
4. [Storage and batch variants](sequence-diagrams/04-storage-and-batch-variants.md)

## Flowcharts

1. [Custom segmentation pipeline flowchart](flowcharts/custom-segmentation-pipeline-flowchart.md)

## Key return contracts

- Initial segmentation requests return immediately with `task_id` and
  `status=processing`.
- Analysis results return asynchronously through `segmentation.result`, then
  become `SegmentationTask.results` in PostgreSQL.
- Frontend live updates come from `/segmentation/stream/{task_id}` SSE.
- Evaluation comparison returns `{comparison: {algorithm: {segments, metrics}}}`
  after all selected task ids have completed.
- Batch evaluation returns `job_id` first, then streams log lines and final
  `{summary, rows, error}` over SSE.
