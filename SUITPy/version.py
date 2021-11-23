# *- encoding: utf-8 -*-
"""
SUITPy version, required package versions, and utilities for checking

@author: maedbhking
based heavily on `nilearn.version`
"""

__version__ = '1.0.4'

_SUITPy_INSTALL_MSG = 'See %s for installation information.' % (
    'https://suitpy.readthedocs.io/en/latest/install.html#installation')

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.16',
        'required_at_installation': True,
        'install_info': _SUITPy_INSTALL_MSG}),
    ('nibabel', {
        'min_version': '2.5',
        'required_at_installation': True}),
    ('pandas', {
        'min_version': '0.24.0',
        'required_at_installation': True,
        'install_info': _SUITPy_INSTALL_MSG}),
    ("requests", {
        "min_version": "2",
        "required_at_installation": True
    }),
    ("scipy", {
        "min_version": "1.0",
        "required_at_installation": True
    })
)

OPTIONAL_MATPLOTLIB_MIN_VERSION = '2.0'


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
            module_name,
            install_info or 'Please install it properly to use SUITPy.')
        exc.args += (user_friendly_info,)
        # Necessary for Python 3 because the repr/str of ImportError
        # objects was changed in Python 3
        if hasattr(exc, 'msg'):
            exc.msg += '. ' + user_friendly_info
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use SUITPy. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_SUITPy_installing=False):
    """Throw an exception if SUITPy dependencies are not installed.

    Parameters
    ----------
    is_SUITPy_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.

    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_SUITPy_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))