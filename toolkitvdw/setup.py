from distutils.core import setup
setup(name="ToolkitVdW",
      version="1.1",
      description="Ignace Van de Woestyne's functions toolkit",
      author="Ignace Van de Woestyne",
      author_email="ignace.vandewoestyne@kuleuven.be",
      py_modules=["toolkitvdw.finance", "toolkitvdw.miscellaneous",
                  "toolkitvdw.optimization", "toolkitvdw.visualization",
                  "toolkitvdw.production_light"]
      )
