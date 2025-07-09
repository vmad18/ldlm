# Reproducible script to build and prepare environment
# However, the installation is based on previously installed miniconda.
# Assuming that, under my aliases
# conda_base ... then bash install.sh

REPO=$(pwd)

# modify the installation path and env name if you want
# INSTALLDIR=${WRKSPC}
# INSTALLDIR=${LUSTRE5}
INSTALLDIR=${VASTUSER}

export UV_CACHE_DIR=$VASTUSER/.cache/uv

ENV_NAME="tuolumne_uv_ldlm"

cd ${INSTALLDIR}

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.12 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install -c conda-forge conda-pack libstdcxx-ng --yes

# Load modules
rocm_version=6.3.0

module load rocm/$rocm_version

######### COMPILE UV PACKAGES ########################

cd "${REPO}"

# since I'm normally used to caching and installing envs under $WRKSPC, a specific nfs drive,
# uv gets mad bc it tries to install the venv inside the repo and these are cross filesystems. eg.:
# warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
# If the cache and target directories are on different filesystems, hardlinking may not be supported.
# If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.

# workaround is currently just to try and get everything, code and env on the same filesystem
pip install uv

# if you run this line standalone remember to set the cache dir like above
uv run --index-strategy=unsafe-best-match main_lvae.py
# what's the uv-y way to dry run this so we get the env built but not execute the program?
# do we need a dummy target script?

# note that since uv is installed in conda, need to activate that first

# can activate the env to use it like so 
# source .venv/bin/activate

cd ${INSTALLDIR}


