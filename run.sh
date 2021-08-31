#!/bin/bash


#Gen3
export REPO=/repo/main
export INPUT_COLLECTION=HSC/runs/RC2/w_2021_30/DM-31182
setup -k -r .
pipetask run -t lsstdesc.descwl.coadd.coadd_in_cells.CoaddInCellsTask -b $REPO --input $INPUT_COLLECTION --output u/$USER/coaddTest -d "skymap='hsc_rings_v1' AND tract=9615 AND patch=45 AND band='r'" --config coaddsInCellsV1:seed=12345678 --register-dataset-types
butler query-datasets $REPO --collections u/$USER/coaddTest/* coaddsInCellsV1
