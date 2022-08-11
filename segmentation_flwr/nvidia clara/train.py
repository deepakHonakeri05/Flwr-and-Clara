import os
from data_splitter import EndoscopyDataSplitter
from endoscopy_learner import ENDOSCOPYLearner


def main():
    if os.path.exists('/tmp/endoscopy_valid/valid.pkl') is False:
        splitter = EndoscopyDataSplitter()
        splitter.split()
    learner = ENDOSCOPYLearner()
    learner.initialize()
    learner.local_train()


if __name__ == "__main__":
    main()
