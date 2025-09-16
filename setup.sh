WORK_DIR=/ws/external
SOURCE_DIR=/ws/external/ai_module/src
WEIGHT_DIR=model_weights

# Download weights
if [ -d "$SOURCE_DIR/$WEIGHT_DIR" ]; then
  echo "✅ $TARGET_NAME already exists at $SOURCE_DIR, skipping download."
else
  echo "📥 $TARGET_NAME not found, downloading..."
  wget --save-cookies "$SOURCE_DIR/cookies.txt" --keep-session-cookies --no-check-certificate \
    "https://urserver.kaist.ac.kr:8148/sharing/JuPH9EI7E" -O /dev/null && \
  wget --load-cookies "$SOURCE_DIR/cookies.txt" --content-disposition -L --no-check-certificate \
    "https://urserver.kaist.ac.kr:8148/fsdownload/JuPH9EI7E/$WEIGHT_DIR.tar.xz" \
    -O "$SOURCE_DIR/$WEIGHT_DIR.tar.xz" && \
  tar -xf "$SOURCE_DIR/model_weights.tar.xz" -C "$SOURCE_DIR" && \
  rm "$SOURCE_DIR/$WEIGHT_DIR.tar.xz" "$SOURCE_DIR/cookies.txt"
  python3 /ws/external/tools/download_hugging_face.py
fi

# Build
cd  ${WORK_DIR}/system/unity
catkin_make

export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src/vlm/scripts
echo 'export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src/vlm/scripts' >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src/task_planner/scripts/task_planner
echo 'export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src/task_planner/scripts/task_planner' >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src
echo 'export PYTHONPATH=$PYTHONPATH:/ws/external/ai_module/src' >> ~/.bashrc

source ~/.bashrc
