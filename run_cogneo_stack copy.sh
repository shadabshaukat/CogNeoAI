#!/bin/bash
# Source environment variables from .env if present
if [ -f .env ]; then
  echo "Sourcing .env for environment variables..."
  set -o allexport
  source .env
  set +o allexport
fi

# Start FastAPI backend (Uvicorn) in background (old port: 8000)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo $FASTAPI_PID > .fastapi_pid

# Start Gradio frontend in background (port: 7866)
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7866
python gradio_app.py &
GRADIO_PID=$!
echo $GRADIO_PID > .gradio_pid

# Start Streamlit frontend in background (force port 8501)
streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > .streamlit_pid

echo
# ===== Fancy ASCII Banner (Rainbow) =====
COLORS=(196 202 208 214 220 226 46 51 27 93 165)
banner="$(cat <<'EOF'
  ████████╗  ██████╗  ██████╗███╗   ██╗██████  ██████╗      █████╗  ██████╗ 
 ██╔══════╝██╔═══██╗██╔════╝ ███╗  ██║██╔════╝██╔═══██╗    ██╔══██╗ ╚═██╔═╝ 
 ██║       ██║   ██║██║  ███╗█╔██╗ ██║█████╗  ██║   ██║    ███████║   ██║   
 ██║       ██║   ██║██║   ██║█║╚██╗██║██╔══╝  ██║   ██║    ██╔══██║   ██║   
 ██║       ╚██████╔╝╚██████╔╝█║ ╚████║███████╗╚██████╔╝    ██║  ██║   ██║   
 ╚███████╗  ╚═════╝  ╚═════╝ ═╝  ╚═══╝╚══════╝ ╚═════╝     ╚═╝  ╚═╝ ██████║
                        COGNEO AI
EOF
)"
i=0
while IFS= read -r line; do
  color="${COLORS[$(( i % ${#COLORS[@]} ))]}"
  printf "\e[38;5;%sm%s\e[0m\n" "$color" "$line"
  i=$((i+1))
done <<< "$banner"
echo

# ===== URLs for Web Stacks =====
HOST_DISPLAY="${COGNEO_HOST_DISPLAY:-localhost}"
printf "\e[1;32mFastAPI:\e[0m   http://%s:8000/health\n" "$HOST_DISPLAY"
printf "\e[1;32mGradio:\e[0m    http://%s:7866\n" "$HOST_DISPLAY"
printf "\e[1;32mStreamlit:\e[0m http://%s:8501\n" "$HOST_DISPLAY"
echo
echo "To stop all, run: bash stop_cogneo_stack.sh"
echo

wait
