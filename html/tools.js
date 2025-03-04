async function load_python(console_node) {
  let pyodide = await loadPyodide();
  // Capture python prints
  pyodide.globals.set("console_node", document.getElementById(console_node));
  pyodide.runPython(`
        import sys
        import js
        class HTMLPrinter:
            def __init__(self, node, stderr=False):
                self.node = node
                # Select preprocessor for node elements
                if stderr:
                    def pre_proc(el):
                        el.setAttribute("class", "stderr")
                        return True
                else:
                    def pre_proc(el):
                        return self.node.scrollTop == self.node.scrollHeight - self.node.clientHeight
                self.pre_proc = pre_proc

            def write(self, message):
                el = js.document.createElement('span')
                el.innerHTML = message
                keep_vis = self.pre_proc(el)
                self.node.appendChild(el)
                if keep_vis:
                    self.node.scrollTop = self.node.scrollHeight

            def flush(self):
                pass

        sys.stdout = HTMLPrinter(console_node)
        sys.stderr = HTMLPrinter(console_node, True)
        console_node.innerHTML = ''
        print('<b>HTMLPrinter started<b/>')
        print('This is a stderr print example', file=sys.stderr)
    `);
  // Show some version info
  await pyodide.loadPackage("numpy");
  await pyodide.loadPackage("sympy");
  pyodide.runPython(`
        import pyodide
        import numpy as np, sympy
        print(f'python: {sys.version}')
        print(f'Pyodide: {pyodide.__version__}')
        print(f'numpy: {np.__version__}')
        print(f'sympy: {sympy.__version__}')
    `);
  // Download our modules from GitHub and run it, block main() execution
  let pythonCode = await (await fetch("../exp_poly_approx.py")).text();
  pyodide.globals.set("__name__", null);
  pyodide.runPython(pythonCode);
  // Table helpers
  pyodide.runPython(`
        arr_append_child = np.vectorize(js.Node.prototype.appendChild.call, otypes=[object])
        arr_elem_set_attribute = np.vectorize(js.Element.prototype.setAttribute.call, otypes=[object])
        @np.vectorize(otypes=[object], excluded={'tag', 'par'})
        def arr_create_elem_with_content(tag, text):
            el = js.document.createElement(tag)
            el.append(text)
            return el
        def append_table_row(table, data, td_tag='td'):
            td = arr_create_elem_with_content(td_tag, data)
            tr = js.document.createElement('tr')
            arr_append_child(tr, td)
            arr_append_child(table, tr)
            return tr, td
    `);
  return pyodide;
}

// Show/hide elements by CSS class
function show_by_class(show_class, hide_classes) {
  for (let i = 0; i < document.styleSheets.length; i++) {
    for (let j = 0; j < document.styleSheets[i].cssRules.length; j++) {
      let cssRule = document.styleSheets[i].cssRules[j];
      if (cssRule.selectorText === show_class) {
        cssRule.style.display = "block";
      } else if (hide_classes.includes(cssRule.selectorText)) {
        cssRule.style.display = "none";
      }
    }
  }
}
