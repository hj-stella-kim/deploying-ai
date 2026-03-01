import gradio as gr
from main import chat

# ============================================================
# Chat Wrapper
# ============================================================
def chat_wrapper(message, history):
    """Processes chat using 'chat' function from main.py"""
    
    # call chat function from main.py
    response = chat(message, history)                
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""

# ============================================================
# Gradio UI Construction
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 900px;}") as demo:
    
    gr.Markdown("# Multi-Tool Assistant")
    gr.Markdown("Ask me anything. This tool is specialized for League of Legends, Pitchfork database based music reviews, or weather!")

    # Chat Area
    with gr.Column():
        chatbot = gr.Chatbot(type="messages", height=500)
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask me anything...",
                show_label=False,
                scale=9
            )
            submit_btn = gr.Button("Send", scale=1)

    # ============================================================
    # Event Bindings
    # ============================================================
    
    # Chat Submission
    msg_input.submit(
        fn=chat_wrapper,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    submit_btn.click(
        fn=chat_wrapper,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)