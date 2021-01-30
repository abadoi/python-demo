import sys
import argparse
import json
import utils as ut 
import joblib
import os

PATTERN_MATCHING_TOOL = os.environ["PATTERN_MATCHING_TOOL"]
ML_MODEL = os.environ["ML_MODEL"]

def main():   
    parser = argparse.ArgumentParser(description='Main point of entry for prediciting vulnerabilities from binaries.')
    # parser.add_argument('-m','--model', help='Machine Learning model used to predict vulnerabilities.', required=True)
    parser.add_argument('-b','--binary', help='Binary file to be analyzed.', required=True)

    args = parser.parse_args()
    file_name = args.binary
    # ml_model = args.model

    if not os.path.exists(file_name):
        print("Binary {} not found.\n".format(file_name))
        return
    else:
        print("Binary {} found.\n".format(file_name))
    
    # set results dir
    out_dir = os.path.join(os.getcwd(), 'out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get retdec ir file
    ut.get_retdec(file_name, file_name + '_retdec')
    retdec_ir = file_name + '_retdec.bc'
    
    #TODO: remove (only for test)
    # retdec_ir = 'data/test/main_retdec.ll'
    
    if os.path.exists(retdec_ir):
        ut.predict(PATTERN_MATCHING_TOOL, retdec_ir, ML_MODEL, out_dir)


if __name__ == "__main__":
    main()