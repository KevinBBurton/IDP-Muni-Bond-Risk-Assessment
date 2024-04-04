from PyPDF2 import PdfReader
import re
import os

from functools import reduce


def process_cusip(file, filter, window_size):
    """
    Takes the CUSIP PDF, attempts to find the risk section by counting occurrences of associated words in a sliding window
    and returns a filtered text
    :param file: Path to the PDF as a String
    :param filter: Document containing all phrases to filter for in the PDF
    :param window_size: Size of the sliding window used to identify the risk section; larger windows dramatically increase computation time
    :return: Text of the risk section, filtered to sentences containing the keywords of the filter file
    """
    filter_words = open(filter).readlines()
    filter_words = [word.removesuffix('\n') for word in filter_words]

    reader = PdfReader(file)

    # this code tries to find the risk section of a given document by sliding a 3-page window over the document and
    # counting the occurrences of the word 'risk'. The page that initiate that window is returned as the one most likely
    # to begin the risk section

    best_page_number = 0
    best_pages_count = 0
    for page_number in range(3,
                             min(len(reader.pages) - window_size, 50)):  # start at page 3 to not get the highest count in the table of contents; end at 90 to not get it in some appendix

        text = reduce(lambda x, y: x + y, [reader.pages[page_number + i].extract_text() for i in range(window_size)])
        text = text.lower()

        current_pages_count = 0
        for word in filter_words:
            current_pages_count += len(re.findall(word, text))

        # greater chosen over greater equals in case of appendices (similar to the reason table of contents are avoided)
        if current_pages_count > best_pages_count:
            best_page_number = page_number
            best_pages_count = current_pages_count

    print(f"Identified risk section starting from page {best_page_number + 1}")

    risk_text = reduce(lambda x, y: x + y,
                       [reader.pages[i].extract_text().lower() for i in
                        range(best_page_number, best_page_number + window_size)])

    sentences = risk_text.split('.')

    filtered_sentences = []
    for sentence in sentences:
        for word in filter_words:
            if re.search(word, sentence) is not None:
                filtered_sentences.append(sentence + ".")
                break

    return reduce(lambda x, y: x + y, filtered_sentences)


def create_training_dataset(cusip_directory, dataset_directory):
    """
    Takes a directory containing CUSIPs labeled by the directory they are contained in, converts them to text files, and
    writes them into a new directory keeping the labels; also adds the length of the file to the end to aid learning
    :param cusip_directory: Path to the directory containing the CUSIP PDFs
    :param dataset_directory: Path to the directory where the converted dataset should be written to
    """
    print("Creating new dataset")

    print(f"Writing input data to {dataset_directory}")

    # the CUSIPS are labeled by the directory they are contained in, therefore when converting them to useful text files the directory structure has to be the same
    for (path, _, files) in os.walk(cusip_directory, topdown=True):
        class_directory = path.removeprefix(cusip_directory) + '/'

        files = list(filter(lambda x: '.pdf' in x or '.PDF' in x, files))  # remove unwanted files

        index = 1
        for file in files:
            print(f"[{class_directory.removesuffix('/')}][{index}/{len(files)}] Converting {file}...")

            risk_text = process_cusip(cusip_directory + class_directory + file, 'filter.txt', 3)

            data_path = dataset_directory + class_directory
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            with open(data_path + file.removesuffix('.pdf') + '.txt', 'w', encoding='utf-8') as f:
                f.write(
                    risk_text + f" length: {len(risk_text)}")  # adding the length of the text to the file so it can be considered by the classifier
            index += 1
