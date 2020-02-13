from IPython.display import HTML


def progress_bar_terminal(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printend="\r"):
    """Call in a loop to create terminal progress bar.

    Parameters
    ----------
    iteration: int
        current iteration
    total: int
        total iterations
    prefix: str
        prefix string
    suffix: str
        suffix string
    decimals: int
        positive number of decimals in percent complete
    length: int
        character length of bar
    fill: str
        bar fill character
    printend: str
        end character (e.g. "\r", "\r\n")
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlength = int(length * iteration // total)
    bar = fill * filledlength + '-' * (length - filledlength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printend)
    # Print New Line on Complete
    if iteration == total:
        print()


def progress_bar_notebook(value, max_value=100):
    result = int((value/max_value)*100)
    return HTML("""
              <p>
                  Progress: {result}% Complete
              <p/>            
              <progress
                  value='{value}'
                  max='{max_value}',
                  style='width: 25%',
              >
                  {value}
              </progress>

    """.format(value=value, max_value=max_value, result=result))