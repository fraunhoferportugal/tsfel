from IPython.display import HTML
from IPython import get_ipython


def progress_bar_terminal(iteration, total, prefix='', suffix='', decimals=0, length=100, fill='█', printend="\r"):
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


def progress_bar_notebook(iteration, total=100):
    """ Progress bar for notebooks.

    Parameters
    ----------
    iteration: int
        current iteration
    total: int
        total iterations

    Returns
    -------
        Progress bar for notebooks

    """
    result = int((iteration/total)*100)
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

    """.format(value=iteration, max_value=total, result=result))


def display_progress_bar(iteration, total, out):
    """ Displays progress bar according to python interface.

    Parameters
    ----------
    iteration: int
        current iteration
    total: int
        total iterations
    out: progress bar notebook output

    """

    if (get_ipython().__class__.__name__ == 'ZMQInteractiveShell') or (
            get_ipython().__class__.__name__ == 'Shell') and out is not None:
        out.update(progress_bar_notebook(iteration + 1, len(total)))
    else:
        progress_bar_terminal(iteration + 1, len(total), prefix='Progress:', suffix='Complete',
                              length=50)
    return
