# descwl-coadd-task
Task to run coaddition in cells

## Setting up the environment at NCSA

Load the shared `lsst-scipipe` environment:

```bash
source /software/lsst/stack/loadLSST.bash
setup lsst_distrib
```

Then stack another shared conda environment with relevant LSSTDESC packages on top of it.

```bash
conda activate --stack /project/kannawad/descrepos/conda/desc-scipipe-20220525-shared
```

Since the `cell_coadds` package is not part of the Rubin Science Pipelines, this needs to be setup manually.

```bash
git clone https://github.com/lsst-dm/cell_coadds.git
cd cell_coadds
setup -k -r .
scons -j 4
```

Similarly, you must setup this package or manually add the parent directory your `PYTHONPATH`.
```bash
cd descwl-coadd-task
setup -k -r .
```

Setup some convenient environment variables:

```bash
export REPO=/repo/dc2
export INPUT_COLLECTION=2.2i/runs/test-med-1/w_2022_18/DM-34608
```

Finally, invoke the `pipetask run` command. Since the cell-based coaddition is not part of any pipeline, the bare task has to be invoked.
To produce the coadd for `tract=3828`, `patch=19` and `band="i"`, run:

```bash
pipetask run --task lsst.cell_coadds.MultipleCellCoaddBuilderTask -b $REPO -i $INPUT_COLLECTION -o u/$USER/coaddTest -d "skymap='DC2_cells_v1' AND tract=3828 AND patch=19 AND band='i'" --config-file multipleCellCoaddBuilder:config/config.py
```

If the output dataset has never been registered with the butler before, then the first time you run this command (and only for the first time), you will need to include `--register-dataset-types` option to the `pipetask run` command.
