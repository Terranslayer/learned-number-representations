# Design and evaluation notes

The retained `symemerge/numcode` package generates synthetic quantity scenes, trains a visual writer and reader, and evaluates several numerical tasks. The `pred` package contains the later model, replay, and iterated-relearning work.

## Separate recognition from transfer

A reader can memorize a code for every quantity it sees. To test transfer, `draw_exposure` selects a stratified teaching subset and the evaluator initializes a fresh reader. It measures performance on quantities outside that teaching subset with separate random streams for teaching and examination.

The learned, random, unary, and positional representations receive the same teaching budget. The control codes are evaluation references; they are not supplied as target symbols for the writer to imitate.

## Keep split definitions explicit

The global quantity generator excludes designated interpolation holes and an extrapolation range from training. The reader-transfer test makes an additional split inside the permitted training domain. A result on the latter should not be described as unrestricted extrapolation.

## Scope of this snapshot

The source is an extract of the later number-code work. The original research workspace also contains earlier architectures, notebooks, and launch machinery. Those are not needed to inspect this experiment. The selected saved results come from the original run; the snapshot does not bundle model checkpoints or reproduce training automatically.

No model behavior was intentionally changed for this presentation. Model validation belongs on the configured Linux CUDA environment. The source still contains experiment-specific settings and Chinese comments where they explain the original work.
