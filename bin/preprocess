import inspect
import argparse
import importlib


MODULE = 'core.data.{}'
PREPROCESS_FN = 'preprocess_{}'
AVAILABLE_DATASETS = {
    'dglrgcn',
    'dortmund'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subparser_name')

    for dataset in AVAILABLE_DATASETS:
        subpar = subparsers.add_parser(dataset)

        # Get preprocessing arguments
        module = importlib.import_module(MODULE.format(dataset))
        preprocess_fn = getattr(module, PREPROCESS_FN.format(dataset))

        specs = inspect.getfullargspec(preprocess_fn)

        defaults = specs.kwonlydefaults

        if defaults is None:
            required_args = specs.kwonlyargs
            optional_args = []
        else:
            required_args = specs.kwonlyargs[:-len(defaults)]
            optional_args = specs.kwonlyargs[-len(defaults):]

        for arg in required_args:
            subpar.add_argument('--' + arg, required=True)

        for arg in optional_args:
            subpar.add_argument('--' + arg, default=defaults[arg])


    args = parser.parse_args()

    print("=== PARSED COMMAND-LINE OPTIONS ===")
    print(args)
    print()

    fn_args = args.__dict__
    dataset = fn_args.pop('subparser_name')

    module = importlib.import_module(MODULE.format(dataset))
    preprocess_fn = getattr(module, PREPROCESS_FN.format(dataset))
    
    print("=== PRINT FROM PREPROCESS_FN ===")
    preprocess_fn(**fn_args)
    print()

    print("=== PREPROCESS_FN USED ===")
    print(inspect.getsource(preprocess_fn))
