# descwl-coadd-task
Task to run coaddition in cells

[![tests](https://github.com/LSSTDESC/descwl-coadd-task/actions/workflows/tests.yaml/badge.svg)](https://github.com/LSSTDESC/descwl-coadd-task/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/LSSTDESC/descwl-coadd-task/graph/badge.svg?token=njZYWcF4pm)](https://codecov.io/gh/LSSTDESC/descwl-coadd-task)

## Setting up the environment at USDF (for Gen3 butler)

Load a custom conda environment that has the Rubin Science Pipelines and other relevant packages installed.

For example,

```bash
conda activate /sdf/home/e/esheldon/miniconda3/envs/stack
```

Regularly update the ``stackvana`` package in the environment.


Similarly, you must setup this package or manually add the parent directory your `PYTHONPATH`.
```bash
cd descwl-coadd-task
setup -k -r .
```

Setup some convenient environment variables:

```bash
export REPO=/sdf/data/rubin/repo/dc2
export INPUT_COLLECTION=2.2i/runs/test-med-1/w_2023_49/DM-42037
```

Finally, invoke the `pipetask run` command.
To produce the coadd for `tract=3828`, `patch=19` and `band="i", "r"`, run:

```bash
pipetask run -p pipeline.yaml -b $REPO -i $INPUT_COLLECTION -o u/$USER/coaddTest -d "skymap='DC2_cells_v1' AND tract=3828 AND patch=19 AND band IN ('i', 'r')"
```

If the output dataset has never been registered with the butler before, then the first time you run this command (and only for the first time), you will need to include `--register-dataset-types` option to the `pipetask run` command.

## Notes on the pipeline file

A minimal pipeline file is provided in `pipeline.yaml`.
This file currently calls the `PipelineTask`s to generate warps and to generate
coadds from the warps.
While bare tasks can be run using the ``--task`` invocation instead of `-p`, it
is recommended to run the pipeline file instead.
The pipeline file enforces 'contracts' that ensure that the different `Task`s
are configured in a consistent manner.
For instance, attempting to warp the PSF with a different kernel than that was
used to warp the images using
```bash
pipetask run -p pipeline.yaml -b $REPO -i $INPUT_COLLECTION -o u/$USER/coaddTest \
-d "skymap='DC2_cells_v1' AND tract=3828 AND patch=19 AND band IN ('i', 'r') \
-c "assembleCoadd:psf_warper.warpingKernelName='lanczos5'"
```
would result in an error.

## Notes on configurability of the `PipelineTask`s

The `PipelineTask`s are highly configurable and can be configured in multiple
places.

1. Default values specified in the definition of the `Config` field.
2. Default values overridden in the `setDefaults` method of the `ConfigClass`.
3. Instrument-specific values in the `obs_*` package.
4. Values specified in the pipeline file.
5. Values specified as a config file with an invocation of `-C`.
6. Values specified on the command-line with an invocation of `-c`.

Because the values can be overridden multiple times, the final set of
configuration values that the `Task`s were run with is stored as an output
itself (which can be reused with the `-C` option).
