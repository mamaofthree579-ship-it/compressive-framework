import streamlit as st
import io
import traceback
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import re

st.set_page_config(page_title="AI Code Fixer + Visualizer", layout="wide")

st.title("🧠 AI Code Fixer + Visualizer")
st.write("Paste your Matplotlib/Numpy plotting code below. If errors occur, the app will auto-diagnose and fix them!")

code_input = st.text_area(
    "Python plotting code:",
    height=300,
    placeholder="Example:\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nx = np.linspace(-5,5,200)\ny = np.sin(x)\nplt.plot(x,y)\nplt.show()"
)

col1, col2 = st.columns([1, 2])
run_button = col1.button("🎨 Run Code")
auto_fix = col2.checkbox("🧩 Auto-fix errors", value=True)


def run_code(code: str):
    """Executes code safely and returns figure, output, and errors."""
    output = io.StringIO()
    fig = None
    error = None

    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            exec_globals = {"np": np, "plt": plt}
            plt.close("all")
            plt.figure()
            exec(code, exec_globals)
            fig = plt.gcf()
        except Exception as e:
            error = traceback.format_exc()

    return fig, output.getvalue(), error


def auto_correct_code(code: str, error: str):
    """Basic fixer for common Python/matplotlib mistakes."""
    fixed = code

    # Common fixes
    if "NameError" in error and "plt" in error and "import matplotlib" not in fixed:
        fixed = "import matplotlib.pyplot as plt\n" + fixed
    if "NameError" in error and "np" in error and "import numpy" not in fixed:
        fixed = "import numpy as np\n" + fixed
    if "SyntaxError" in error and not fixed.strip().endswith("plt.show()"):
        fixed += "\nplt.show()"
    if "ValueError" in error and "x and y" in error:
        fixed = re.sub(r"plt\.plot\([^)]*\)", "plt.plot(np.arange(10), np.arange(10))", fixed)
    if "plt.show" not in fixed:
        fixed += "\nplt.show()"

    return fixed


if run_button and code_input.strip():
    with st.spinner("Running your code..."):
        fig, logs, err = run_code(code_input)

    if err and auto_fix:
        st.warning("⚠️ Error detected — attempting auto-fix...")
        st.text_area("Error Traceback:", err, height=200)

        fixed_code = auto_correct_code(code_input, err)
        st.code(fixed_code, language="python")

        with st.spinner("Applying fixes and re-running..."):
            fig_fixed, logs_fixed, err_fixed = run_code(fixed_code)

        if err_fixed:
            st.error("❌ Fix attempt failed.")
            st.text_area("Remaining Error:", err_fixed, height=200)
        else:
            st.success("✅ Code fixed successfully!")
            st.pyplot(fig_fixed)
            st.download_button("📥 Download Fixed Plot", fig_fixed.canvas.tostring_rgb(), "fixed_plot.png")

    elif err:
        st.error("❌ Error running code.")
        st.text_area("Error Traceback:", err, height=200)
    else:
        st.success("✅ Code executed successfully!")
        st.pyplot(fig)
        st.download_button("📥 Download Image", fig.canvas.tostring_rgb(), "plot.png")

else:
    st.info("👆 Paste plotting code and click **Run Code**.")
