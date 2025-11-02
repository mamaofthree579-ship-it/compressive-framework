import streamlit as st
import io
import contextlib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Code to Visual Generator", layout="wide")

st.title("🧠 Code → Visual Generator")
st.write("Paste Python plotting code below (e.g. Matplotlib + NumPy). Click **Generate Visual** to render the image.")

code_input = st.text_area(
    "Python code:",
    height=300,
    placeholder="Example:\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nx = np.linspace(-5,5,200)\ny = np.sin(x)\nplt.plot(x,y)\nplt.show()"
)

col1, col2 = st.columns([1, 2])

with col1:
    run_button = st.button("🎨 Generate Visual")

if run_button and code_input.strip():
    try:
        # Redirect stdout/stderr to capture print outputs
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            # Fresh environment for execution
            exec_globals = {"np": np, "plt": plt}
            plt.figure()  # ensure new figure
            exec(code_input, exec_globals)

        # Get current figure
        fig = plt.gcf()
        st.pyplot(fig)

        # Save to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="📥 Download Image (PNG)",
            data=buf,
            file_name="generated_visual.png",
            mime="image/png"
        )

        logs = output.getvalue()
        if logs:
            st.text_area("Execution log:", logs, height=150)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Paste your plotting code and click **Generate Visual**.")
