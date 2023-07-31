import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lego_image_converter',
    packages=['lego_image_converter'],
    version='0.1.2',
    license='MIT',
    description='minor update after formal release',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Zequn Li',
    author_email='zequn1992@gmail.com',
    url='https://github.com/zl3311/lego_image_converter',
    install_requires=['requests', 'numpy', 'matplotlib', 'pillow', 'basic_colormath'],
    keywords=["pypi", "lego_image_converter", "lego", "image", "8bit", "art"],
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    download_url="https://github.com/zl3311/lego_image_converter/archive/refs/tags/0.1.2.tar.gz",
)
