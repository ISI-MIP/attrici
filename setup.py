from setuptools import setup

setup(
    name="isi-cfact",
    version="0.1",
    description="Counterfactual climate data from past datasets for the ISIMIP project.",
    url="http://github.com/isi-mip/isi-cfact",
    author="Matthias Mengel, Benjamin Schmidt",
    author_email="matthias.mengel@pik-potsdam.de",
    license="GPLv3",
    packages=["idetrend"],
    zip_safe=False,
)
