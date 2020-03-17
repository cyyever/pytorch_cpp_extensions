from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="pytorch_cpp",
    ext_modules=[
        cpp_extension.CppExtension(
            "pytorch_cpp",
            ["src/synced_tensor_dict.cpp"])],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension},
)
