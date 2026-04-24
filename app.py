import streamlit as st
from search import search
from PIL import Image

st.title("🎥 Intelligent Video Search Engine")

query = st.text_input("Enter your search query")

if st.button("Search"):
    results = search(query, top_k=5)

    if not results:
        st.warning("No results found. Try a different query.")
    else:
        for idx in range(0, len(results), 2):
            row_results = results[idx:idx + 2]
            cols = st.columns(2)

            for i, result in enumerate(row_results):
                with cols[i]:
                    st.markdown(
                        "<div style='border:1px solid #e6e6e6; border-radius:16px; padding:16px; margin-bottom:16px; background-color:#fafafa;'>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### Result {idx + i + 1}")
                    st.image(Image.open(result["frame"]), use_column_width=True)
                    st.markdown(f"**Score:** {result['score']:.4f}")
                    st.markdown(
                        f"**Timestamp:** <span style='color:#ff4b4b; font-weight:bold;'>{result.get('timestamp', 'N/A')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(result["frame"])
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
