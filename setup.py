from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

setup(
    name="icounter",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Counterfactual climate data from past datasets for the ISIMIP project.",
    url="http://github.com/isi-mip/isi-cfact",
    author="Matthias Mengel, Benjamin Schmidt",
    author_email="matthias.mengel@pik-potsdam.de",
    license="GPLv3",
    packages=["icounter"],
    zip_safe=False,
)
