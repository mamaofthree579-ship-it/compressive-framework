import streamlit as st
import io
import traceback
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import re

st.set_page_config(page_title="AI Code Fixer + Visualizer", layout="wide")

st.title("🧩 AI Code Fixer + Visualizer v2")
st.write("Paste your Python plotting code. The app will detect and **auto-correct** missing variables or imports.")

code_input = st.text_area(
    "Python plotting code:",
    height=300,
    placeholder="Example:\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nfields = ['Graviton','Chronon','Cognon']\nvalues = [0.7,0.5,0.6]\nplt.bar(fields, values)\nplt.show()"
)

col1, col2 = st.columns([1, 2])
run_button = col1.button("🎨 Run Code")
auto_fix = col2.checkbox("🧠 Auto-fix errors", value=True)


def run_code(code: str):
    """Executes user code safely and returns the figure, output logs, and any error."""
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
        except Exception:
            error = traceback.format_exc()

    return fig, output.getvalue(), error


def auto_correct_code(code: str, error: str):
    """Auto-fix common issues in matplotlib/numpy plotting code."""
    fixed = code.strip()

    # Fix missing imports
    if "NameError" in error and "plt" in error and "import matplotlib" not in fixed:
        fixed = "import matplotlib.pyplot as plt\n" + fixed
    if "NameError" in error and "np" in error and "import numpy" not in fixed:
        fixed = "import numpy as np\n" + fixed

    # Fix missing plt.show()
    if "plt.show" not in fixed:
        fixed += "\nplt.show()"

    # Detect undefined variable and inject placeholder data
    undefined_vars = re.findall(r"NameError: name '(\w+)' is not defined", error)
    for var in undefined_vars:
        if var == "fields":
            st.info(f"⚙️ Auto-defining missing variable `{var}` = ['A','B','C']")
            fixed = f"{var} = ['A','B','C']\n" + fixed
        elif var == "values" or var == "curvatures":
            st.info(f"⚙️ Auto-defining missing variable `{var}` = [1,2,3]")
            fixed = f"{var} = [1, 2, 3]\n" + fixed
        elif var == "x":
            st.info(f"⚙️ Auto-defining missing variable `{var}` = np.linspace(-5,5,100)")
            fixed = f"{var} = np.linspace(-5,5,100)\n" + fixed
        elif var == "y":
            st.info(f"⚙️ Auto-defining missing variable `{var}` = np.sin(x)")
            fixed = f"{var} = np.sin(x)\n" + fixed
        else:
            st.info(f"⚙️ Auto-defining missing variable `{var}` = 1.0")
            fixed = f"{var} = 1.0\n" + fixed

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

            # ✅ Convert plot to PNG buffer for download
            img_bytes = io.BytesIO()
            fig_fixed.savefig(img_bytes, format="png", bbox_inches="tight")
            img_bytes.seek(0)

            st.download_button(
                "📥 Download Fixed Plot",
                data=img_bytes,
                file_name="fixed_plot.png",
                mime="image/png"
            )

    elif err:
        st.error("❌ Error running code.")
        st.text_area("Error Traceback:", err, height=200)
    else:
        st.success("✅ Code executed successfully!")
        st.pyplot(fig)

        # ✅ Convert plot to PNG buffer for download
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png", bbox_inches="tight")
        img_bytes.seek(0)

        st.download_button(
            "📥 Download Image",
            data=img_bytes,
            file_name="plot.png",
            mime="image/png"
        )

else:
    st.info("👆 Paste plotting code and click **Run Code**.")
