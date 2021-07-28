#!/bin/bash


#Gen3
export REPO=/project/hsc/gen3repo/rc2v21_0_0_rc1_ssw48
setup -j -r .
pipetask run -t lsstdesc.pipe.task.coadd_in_cells.CoaddInCellsTask -b $REPO --input HSC/runs/RC2/v21_0_0_rc1 --output u/$USER/coaddTest -d "skymap='hsc_rings_v1' AND tract=9615 AND patch=45" --register-dataset-types
butler query-datasets $REPO --collections u/$USER/* coaddObj
