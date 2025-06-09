"""Example of converting I3-files to SQLite and Parquet."""

import os
from glob import glob

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3TruthExtractor,
    I3FeatureExtractorIceCube86
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

from I3Tray import I3Tray  # pyright: reportMissingImports=false


ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_pone(backend: str, qtop: bool) -> None:
    """Convert p-one I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    ## yes it is shitty i am in a rush - vp 01.06.25

    inputs = [f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/test"]
    # inputs = [f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/k40"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone"
    print("outdir: ", outdir)
    print('inputs: ', inputs)
    gcd_rescue = glob(
        f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/*GCD*"
    )
    if not gcd_rescue:
        print("No GeoCalib files found for p-one.")
        return
    gcd_rescue = gcd_rescue[0]
    print("gcd type", type(gcd_rescue))
    pulsemap_name = "PMTPulseMap"  #PMTResponse_nonoise, K40PulseMap, PMTPulseMap
    converter = CONVERTER_CLASS[backend](
        extractors=[
            I3FeatureExtractorIceCube86(pulsemap_name),
            I3TruthExtractor(),  # Add own truthextractor
            
        ],
        outdir=outdir,
        gcd_rescue=gcd_rescue,
        workers=1,
    )

    if qtop:
        tray = I3Tray()
        ## yes it is shitty i am in a rush - vp 01.06.25
        tray.Add("I3Reader", "reader", FilenameList=[f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/GenerateSingleMuons_39_pmtsim.i3.zst"])
        # tray.Add("I3Reader", "reader", FilenameList=[f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/k40/k40_1.i3.gz"])
        tray.Add("I3NullSplitter", 'splitme', SubEventStreamName="InIceSplit")
        tray.Add("I3Writer", "writer", Filename=f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/GenerateSingleMuons_39_pmtsim_pframe.i3.zst")
        # tray.Add("I3Writer", "writer", Filename=f"{TEST_DATA_DIR}/i3/pone-GenerateSingleMuons_39_10String_7Cluster/k40/k40_1_pframe.i3.zst")
        tray.Execute()
        tray.Finish()
        print("QTOP done")
    else:
        converter(inputs)
        if backend == "sqlite":
            converter.merge_files()

if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Convert I3 files to an intermediate format.
"""
        )

        parser.add_argument("backend", choices=["sqlite", "parquet"])
        # STEP 1 -- if you are using a file that is only Q frames
        parser.add_argument("--qtop", action="store_true", help="Convert I3 inputs into new I3 files containing only P-frames.")
        # parser.add_argument("outdir", default=f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone")
        args, unknown = parser.parse_known_args()
        

        # Run example script
        main_pone(args.backend, args.qtop)


