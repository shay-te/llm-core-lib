import os

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

with open('README.md', 'r') as fh:
    long_description = fh.read()

    setup(
        name='llm_core_lib',
        version='0.1.0',
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
