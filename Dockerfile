from nvcr.io/nvidia/pytorch:24.03-py3

RUN pip install torch_geometric

# Optional dependencies:
RUN pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
RUN pip install --verbose torch_scatter
RUN pip install --verbose torch_sparse
RUN pip install --verbose torch_cluster
RUN pip install --verbose torch_spline_conv

RUN pip install dpu_utils rdkit more_itertools py-repo-root wandb lightning dill ray

# Install VSCode:
RUN wget https://update.code.visualstudio.com/commit:e170252f762678dec6ca2cc69aba1570769a5d39/server-linux-x64/stable
RUN tar -xvf stable
RUN mkdir -p ~/.vscode-server/bin/e170252f762678dec6ca2cc69aba1570769a5d39
RUN mv vscode-server-linux-x64/* ~/.vscode-server/bin/e170252f762678dec6ca2cc69aba1570769a5d39