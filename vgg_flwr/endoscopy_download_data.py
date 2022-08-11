from endoscopy_data_splitter import EndoscopyDataSplitter
from nvflare.apis.fl_context import FLContext


def main():
    splitter = EndoscopyDataSplitter(num_sites=2)
    splitter.split(fl_ctx=FLContext())


if __name__ == "__main__":
    main()
