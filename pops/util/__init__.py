from typing import Optional
import numpy
import rpy2.robjects
import rpy2.robjects.numpy2ri

def save_R(filename: str, numpy_data: Optional[dict[str, numpy.ndarray]] = None, data: Optional[dict[str, any]] = None):
    """ Save data in RDS format

    Args:
        filename (str): Filename to save to
        numpy_data (Optional[dict[str, numpy.ndarray]], optional): Dictionary of numpy data. Defaults to None.
        data (Optional[dict[str, any]], optional): Dictionary of primitive data. Defaults to None.
    """
    all_keys = []
    if numpy_data is not None:
        all_keys = all_keys + list(numpy_data.keys())
        for key in numpy_data.keys():
            value = rpy2.robjects.numpy2ri.numpy2rpy(numpy_data[key])
            rpy2.robjects.r.assign(key, value)
    if data is not None:
        all_keys = all_keys + list(data.keys())
        for key in data.keys():
            rpy2.robjects.r.assign(key, data[key])
    save_command = 'saveRDS(list({}), file="{}")'.format(", ".join(['{0} = {0}'.format(x) for x in all_keys]), filename)
    print(save_command)
    rpy2.robjects.r(save_command)


def read_R(filename: str) -> dict[str, numpy.ndarray]:
    """ Read data from RDS format

    Args:
        filename (str): File to read

    Returns:
        dict[str, numpy.ndarray]: Dictionary of read data
    """
    loaded_data = rpy2.robjects.r['readRDS'](filename)
    if not isinstance(loaded_data, rpy2.robjects.vectors.ListVector):
        return numpy.array(loaded_data)
    return dict(zip(loaded_data.names, map(numpy.array, list(loaded_data))))