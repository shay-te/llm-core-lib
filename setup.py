import os
import re

import setuptools

# llm_core_lib_import

from setuptools import find_namespace_packages, setup
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

dir_path = os.path.dirname(os.path.realpath(__file__))
install_reqs = parse_requirements(os.path.join(dir_path, 'requirements.txt'), session=PipSession)
requirements = []
try:
    requirements = [str(ir.req) for ir in install_reqs]
except:
    requirements = [str(ir.requirement) for ir in install_reqs]

packages1 = setuptools.find_packages()
packages2 = find_namespace_packages(include=['hydra_plugins.*'])
packages = list(set(packages1 + packages2))


def _read_version() -> str:
    # Single source of truth for the version is ``llm_core_lib.__version__``.
    # Regex-extract it (rather than importing the package) so setup.py
    # doesn't need ``core-lib`` available at install time.
    init_path = os.path.join(dir_path, 'llm_core_lib', '__init__.py')
    with open(init_path, 'r') as f:
        match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.M)
    if not match:
        raise RuntimeError(f'cannot find __version__ in {init_path}')
    return match.group(1)


with open('README.md', 'r') as fh:
    long_description = fh.read()

    setup(
        name='llm_core_lib',
        version=_read_version(),
        author='llm_full_name',
        author_email='llm_email',
        description='Shared LLM provider abstraction (OpenAI / Anthropic / Bedrock)',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='llm_url',
        packages=packages,
        license='MIT',
        # llm_classifiers
        install_requires=requirements,
        extras_require={
            'openai': ['openai'],
            'anthropic': ['anthropic'],
            'bedrock': ['boto3'],
            'all': ['openai', 'anthropic', 'boto3'],
        },
        include_package_data=True,
        python_requires='>=3.8',
    )
