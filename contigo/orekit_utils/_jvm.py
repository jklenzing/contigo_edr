import importlib.resources
import orekit


def start_jvm():

    if orekit.getVMEnv() is not None:
        return

    jar_path = (
        importlib.resources
        .files("contigo.orekit_utils")
        / "java"
        / "orekit-utils-1.0.0.jar"
    )

    orekit.initVM(additional_classpaths=[str(jar_path)])