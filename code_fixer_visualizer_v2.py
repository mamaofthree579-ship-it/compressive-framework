import streamlit as st
import io
import traceback
import contextlib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utility: Run code safely and capture ALL figures
# -----------------------------
def run_code(code: str):
    """Safely execute matplotlib code and capture all figures + logs + errors."""
    buffer = io.StringIO()
    exec_globals = {"np": np, "plt": plt}
    plt.close('all')  # clear old figures

    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, exec_globals)
        logs = buffer.getvalue()
        figures = [plt.figure(i) for i in plt.get_fignums()]
        return figures, logs, None
    except Exception:
        err = traceback.format_exc()
        return [], buffer.getvalue(), err
    finally:
        plt.close('all')

# -----------------------------
# Utility: Attempt auto-correction
# -----------------------------
def auto_correct_code(code: str, error_msg: str) -> str:
    """Simple rule-based correction system for missing imports or undefined vars."""
    fixed_code = code

    if "NameError" in error_msg:
        if "np" in error_msg and "import numpy as np" not in fixed_code:
            fixed_code = "import numpy as np\n" + fixed_code
        if "plt" in error_msg and "import matplotlib.pyplot as plt" not in fixed_code:
            fixed_code = "import matplotlib.pyplot as plt\n" + fixed_code
        if "fields" in error_msg and "fields" not in fixed_code:
            fixed_code = "fields = ['A', 'B', 'C']\n" + fixed_code

    if "plt.show" not in fixed_code:
        fixed_code += "\nplt.show()"

    return fixed_code

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Code Fixer & Visualizer", layout="wide")
st.title("🧩 Code Fixer & Visualizer v3")
st.markdown("Paste **Matplotlib/Numpy** code, visualize it, auto-fix errors, and download results.")

code_input = st.text_area("💻 Paste your Python plotting code:", height=300, placeholder="import numpy as np\nimport matplotlib.pyplot as plt\n\n# Your plotting code...")

col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button("▶️ Run Code")
with col2:
    auto_fix = st.checkbox("🛠️ Auto-fix if error", value=True)

if run_button and code_input.strip():
    with st.spinner("Running your code..."):
        figs, logs, err = run_code(code_input)

    if err and auto_fix:
        st.warning("⚠️ Error detected — attempting auto-fix...")
        st.text_area("Error Traceback:", err, height=200)

        fixed_code = auto_correct_code(code_input, err)
        st.code(fixed_code, language="python")

        with st.spinner("Applying fixes and re-running..."):
            figs_fixed, logs_fixed, err_fixed = run_code(fixed_code)

        if err_fixed:
            st.error("❌ Fix attempt failed.")
            st.text_area("Remaining Error:", err_fixed, height=200)
        else:
            st.success("✅ Code fixed successfully!")
            for fig in figs_fixed:
                st.pyplot(fig, clear_figure=False)

            # Download fixed code
            st.download_button(
                "💾 Download Fixed Code (.py)",
                data=fixed_code,
                file_name="fixed_code.py",
                mime="text/x-python"
            )

    elif err:
        st.error("❌ Error running code.")
        st.text_area("Error Traceback:", err, height=200)
    else:
        if figs:
            st.success("✅ Code executed successfully!")
            for fig in figs:
                st.pyplot(fig, clear_figure=False)
        else:
            st.info("⚙️ Code ran successfully but no figures were created.")

        # Download original code
        st.download_button(
            "💾 Download Code (.py)",
            data=code_input,
            file_name="original_code.py",
            mime="text/x-python"
        )

else:
    st.info("👆 Paste your code and click **Run Code** to visualize and debug.")
