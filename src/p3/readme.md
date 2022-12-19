pygraphviz is sometimes annoying to install, especially on windows. You may need to install the visual C++ build tools, then install with

python -m pip install --global-option=build_ext `
              --global-option="-IC:\Program Files\Graphviz\include" `
              --global-option="-LC:\Program Files\Graphviz\lib" `
              pygraphviz

see https://pygraphviz.github.io/documentation/stable/install.html
https://pygraphviz.github.io/