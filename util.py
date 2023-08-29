import plotly.graph_objects as go
import streamlit as st
import base64


def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def visualize(image, bboxes):
    
    width, height = image.size
    shapes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Convert bounding box coordinates to the format expected by Plotly
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=height - y2,
            x1=x2,
            y1=height - y1,
            line=dict(color='red', width=6),
        ))

    fig = go.Figure()

    # Add the image as a layout image
    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch"
        )]
    )

    # Set the axis ranges and disable axis labels
    fig.update_xaxes(range=[0, width], showticklabels=False)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1,
                     range=[0, width], showticklabels=False)

    fig.update_layout(
        height=800,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=10, t=10),
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(label="Original",
                         method="relayout",
                         args=["shapes", []]),
                    dict(label="Detections",
                         method="relayout",
                         args=["shapes", shapes])
                     ],
            )
        ]
    )

    st.plotly_chart(fig)