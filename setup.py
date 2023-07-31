import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lego_image_converter',  # should match the package folder
    packages=['lego_image_converter'],  # should match the package folder
    version='0.0.1',  # important for updates
    license='MIT',  # should match your chosen license
    description='Testing installation of Package',
    long_description=long_description,  # loads your README.md
    long_description_content_type="text/markdown",  # README.md is of type 'markdown'
    author='Zequn Li',
    author_email='zequn1992@gmail.com',
    url='https://github.com/mike-huls/toolbox_public',
    install_requires=['requests', 'numpy', 'matplotlib', 'PIL', 'basic_colormath'],  # list all packages that your package uses
    keywords=["pypi", "lego_image_converter", "Lego", "Image", "8bit", "Art"],  # descriptive meta-data
    classifiers=[  # https://pypi.org/classifiers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    download_url="",
)
