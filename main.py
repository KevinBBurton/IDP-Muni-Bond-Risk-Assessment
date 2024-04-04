import preprocess
import classifier

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to filter out unnecessary tensorflow logs

if __name__ == '__main__':
    cmd = argparse.ArgumentParser()
    cmd.add_argument("-t", "--training",
                     help="Set if a new model should be trained and saved. Expects a path to the directory containing the preprocessed CUSIPs as text files. If -n is set this is the path where the text files should be processed to.")
    cmd.add_argument("-n", "--new_dataset",
                     help="Set together with -t if a new dataset should be processed. Expects a path to the directory containing CUSIPs as PDFs, seperated into directories 'training' and 'testing', and further seperated by labels.")
    cmd.add_argument("-a", "--assessment",
                     help="Perform an assessment on the given CUSIP PDF.")
    cmd.add_argument("-s", "--state",
                     help="State of the CUSIP that is being assessed by -a, to provide a national risk score and rating. Both state and county have to be provided.")
    cmd.add_argument("-c", "--county",
                     help="County of the CUSIP that is being assessed by -a, to provide a national risk score and rating. Both state and county have to be provided.")
    cmd.add_argument("-e", "--evaluation",
                     help="Perform full evaluation using the testing dataset created by -n. Expects a path to the directory containing the proprocessed CUSIPs.")

    args = cmd.parse_args()

    if args.training is not None:
        dataset_directory = args.training

        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        if args.new_dataset is not None:
            cusip_directory = args.new_dataset
            preprocess.create_training_dataset(cusip_directory, dataset_directory)
            train_ds, val_ds, test_ds = classifier.load_dataset(dataset_directory)
            classifier.train_model(train_ds, val_ds, test_ds)
        else:
            train_ds, val_ds, test_ds = classifier.load_dataset(dataset_directory)
            classifier.train_model(train_ds, val_ds, test_ds)

    if args.assessment is not None:
        county = args.county
        state = args.state
        print(f"Performing risk assessment on {args.assessment}")
        classifier.perform_assessment(args.assessment, county, state)

    if args.evaluation is not None:
        dataset_directory = args.evaluation
        classifier.evaluate_model(dataset_directory + "/testing/")

