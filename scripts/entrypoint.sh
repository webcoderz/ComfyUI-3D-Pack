#!/bin/bash


download_hf_model() {
    # Extracts the model name from the URL
    model_name=$(basename $1)
    target_directory=$2

    # Check if the target directory does not exist and create it
    if [ ! -d "$target_directory" ]; then
        mkdir  "$target_directory"
    fi

    # Check if the model already exists in the target directory
    if [ -e "$target_directory/$model_name" ]; then
        echo "Model $model_name already exists in $target_directory."
    else
        # Download the model using wget or curl
        echo "Downloading $model_name to $target_directory..."
        if command -v wget > /dev/null; then
            wget -O "$target_directory/$model_name" "$1"
        elif command -v curl > /dev/null; then
            curl -o "$target_directory/$model_name" "$1"
        else
            echo "Error: Neither wget nor curl is installed."
            return 1
        fi
    fi
}

download_civitai_model() {
    local url="${1}"
    local destination="${2}"
    local user_agent_string="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    echo "Downloading from ${url}, please wait..."

    # Check if the destination directory does not exist and create it
    if [ ! -d "${destination}" ]; then
        mkdir -p "${destination}"
    fi

    # Change to the destination directory
    cd "${destination}"

    # Download the file using curl with the specified user agent
    curl -JsL --remote-name -A "${user_agent_string}" "${url}"

    if [ $? -ne 0 ]; then
        echo "Failed to download from ${url}."
        return 1
    else
        echo "Download completed successfully."
    fi

    # Change back to the original directory (optional, based on your use case)
}


clone_or_update_repo_and_install_requirements() {
    local repo_url="${1}"
    local target_directory="${2}"
    local original_directory=$(pwd)  # Store the current directory

    # Ensure the target directory is an absolute path
    if [[ ! "$target_directory" = /* ]]; then
        target_directory="$original_directory/$target_directory"
    fi

    # Check if the target directory already exists
    if [ ! -d "$target_directory" ]; then
        echo "Cloning repository into $target_directory..."
        # Clone the repository with the specified parameters
        git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules "$repo_url" "$target_directory"
        if [ $? -ne 0 ]; then
            echo "Failed to clone repository."
            cd "$original_directory"
            return 1
        fi
    else
        # Change to the target directory and pull the latest changes
        echo "Repository already exists, updating..."
        cd "$target_directory" && git pull
        if [ $? -ne 0 ]; then
            echo "Failed to update repository."
            cd "$original_directory"
            return 1
        fi
    fi

    # Change to the target directory
    cd "$target_directory"

    # Install Python dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies from requirements.txt..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install Python dependencies."
            cd "$original_directory"
            return 1
        fi
    else
        echo "No requirements.txt found."
    fi

    # Change back to the original directory
    cd "$original_directory"
    echo "Repository setup and dependencies installed successfully."
}

# Stable Cascade
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/comfyui_checkpoints/stable_cascade_stage_c.safetensors" "models/checkpoints"
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/comfyui_checkpoints/stable_cascade_stage_b.safetensors" "models/checkpoints"
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/canny.safetensors" "models/controlnet"
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/inpainting.safetensors" "models/controlnet"
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/super_resolution.safetensors" "models/controlnet"

#VAE
download_hf_model "https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/super_resolution.safetensors" "models/vae"
download_hf_model "https://huggingface.co/madebyollin/taesd/resolve/main/taesd_decoder.safetensors" "models/vae_approx"
download_hf_model "https://huggingface.co/madebyollin/taesdxl/resolve/main/taesdxl_decoder.safetensors" "models/vae_approx"

#upscale
download_hf_model "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" "models/upscale_models"
download_hf_model "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" "models/upscale_models"
download_hf_model "https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth" "models/upscale_models"
download_hf_model "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" "models/upscale_models"
download_hf_model "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth" "models/upscale_models"
download_hf_model "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth" "models/upscale_models"

#embeddings
download_hf_model "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors" "models/embeddings"
download_hf_model "https://huggingface.co/lenML/DeepNegative/resolve/main/NG_DeepNegative_V1_75T.pt" "models/embeddings"

#CLIP Vision
download_hf_model "https://huggingface.co/openai/clip-vit-largeatch14/resolve/main/model.safetensors" "models/clip_vision"
download_hf_model "https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors" "models/clip_vision"

#unCLIP
download_hf_model "https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip-small/resolve/main/image_encoder/model.safetensors" "models/checkpoints"

#controlnet
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth" "models/controlnet"
download_hf_model "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth" "models/controlnet"


# Control-LoRA
download_hf_model "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors" "models/controlnet"
download_hf_model "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors" "models/controlnet"
download_hf_model "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors" "models/controlnet"
download_hf_model "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors" "models/controlnet"




echo "All models downloaded successfully."


echo "downloading custom nodes"



#cd custom_nodes

clone_or_update_repo_and_install_requirements "https://github.com/ltdrdata/ComfyUI-Manager.git" "custom_nodes/ComfyUI-Manager"
#clone_or_update_repo_and_install_requirements "https://github.com/ltdrdata/ComfyUI-Impact-Pack"  "custom_nodes/ComfyUI-Impact-Pack" 
clone_or_update_repo_and_install_requirements "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"  "custom_nodes/ComfyUI-Inspire-Pack" 
clone_or_update_repo_and_install_requirements "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"  "custom_nodes/ComfyUI-Frame-Interpolation" 
clone_or_update_repo_and_install_requirements "https://github.com/Fannovel16/ComfyUI-Video-Matting"  "custom_nodes/ComfyUI-Video-Matting" 
clone_or_update_repo_and_install_requirements "https://github.com/BlenderNeko/ComfyUI_Cutoff" "custom_nodes/ComfyUI_Cutoff"
clone_or_update_repo_and_install_requirements "https://github.com/WASasquatch/PPF_Noise_ComfyUI"  "custom_nodes/PPF_Noise_ComfyUI" 
clone_or_update_repo_and_install_requirements "https://github.com/WASasquatch/PowerNoiseSuite"  "custom_nodes/PowerNoiseSuite" 
clone_or_update_repo_and_install_requirements "https://github.com/Jordach/comfylasma" "custom_nodes/comfylasma"
clone_or_update_repo_and_install_requirements "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes" "custom_nodes/ComfyUI_Comfyroll_CustomNodes"
clone_or_update_repo_and_install_requirements "https://github.com/space-nuko/ComfyUI-OpenPose-Editor" "custom_nodes/ComfyUI-OpenPose-Editor"
clone_or_update_repo_and_install_requirements "https://github.com/twri/sdxl_prompt_styler" "custom_nodes/sdxl_prompt_styler"
clone_or_update_repo_and_install_requirements "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved" "custom_nodes/ComfyUI-AnimateDiff-Evolved"
clone_or_update_repo_and_install_requirements "https://github.com/AIrjen/OneButtonPrompt" "custom_nodes/OneButtonPrompt"
clone_or_update_repo_and_install_requirements "https://github.com/WASasquatch/was-node-suite-comfyui"  "custom_nodes/was-node-suite-comfyui" 
clone_or_update_repo_and_install_requirements "https://github.com/cubiq/ComfyUI_essentials" "custom_nodes/ComfyUI_essentials"
clone_or_update_repo_and_install_requirements "https://github.com/crystian/ComfyUI-Crystools"  "custom_nodes/ComfyUI-Crystools" 
clone_or_update_repo_and_install_requirements "https://github.com/ssitu/ComfyUI_UltimateSDUpscale" "custom_nodes/ComfyUI_UltimateSDUpscale"
clone_or_update_repo_and_install_requirements "https://github.com/gokayfem/ComfyUI_VLM_nodes"  "custom_nodes/ComfyUI_VLM_nodes" 
clone_or_update_repo_and_install_requirements "https://github.com/Fannovel16/comfyui_controlnet_aux"  "custom_nodes/comfyui_controlnet_aux" 
clone_or_update_repo_and_install_requirements "https://github.com/Stability-AI/stability-ComfyUI-nodes"  "custom_nodes/stability-ComfyUI-nodes"  
clone_or_update_repo_and_install_requirements "https://github.com/jags111/efficiency-nodes-comfyui"  "custom_nodes/efficiency-nodes-comfyui"  
clone_or_update_repo_and_install_requirements "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"  "custom_nodes/ComfyUI-VideoHelperSuit"e  
clone_or_update_repo_and_install_requirements "https://github.com/pythongosssss/ComfyUI-Custom-Scripts" "custom_nodes/ComfyUI-Custom-Scripts"
clone_or_update_repo_and_install_requirements "https://github.com/WASasquatch/FreeU_Advanced" "custom_nodes/FreeU_Advanced"
clone_or_update_repo_and_install_requirements "https://github.com/city96/SD-Advanced-Noise" "custom_nodes/SD-Advanced-Noise"
clone_or_update_repo_and_install_requirements "https://github.com/kadirnar/ComfyUI_Custom_Nodes_AlekPet" "custom_nodes/ComfyUI_Custom_Nodes_AlekPet"
clone_or_update_repo_and_install_requirements "https://github.com/sipherxyz/comfyui-art-venture"  "custom_nodes/comfyui-art-venture"  
clone_or_update_repo_and_install_requirements "https://github.com/evanspearman/ComfyMath"  "custom_nodes/ComfyMath"  
clone_or_update_repo_and_install_requirements "https://github.com/Gourieff/comfyui-reactor-node"  "custom_nodes/comfyui-reactor-node"  
clone_or_update_repo_and_install_requirements "https://github.com/rgthree/rgthree-comfy"  "custom_nodes/rgthree-comfy"  
clone_or_update_repo_and_install_requirements "https://github.com/giriss/comfy-image-saver"  "custom_nodes/comfy-image-saver"  
clone_or_update_repo_and_install_requirements "https://github.com/gokayfem/ComfyUI-Depth-Visualization"  "custom_nodes/ComfyUI-Depth-Visualization" 
clone_or_update_repo_and_install_requirements "https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI" "custom_nodes/ControlNet-LLLite-ComfyUI"
clone_or_update_repo_and_install_requirements "https://github.com/gokayfem/ComfyUI-Dream-Interpreter"  "custom_nodes/ComfyUI-Dream-Interpreter" 
clone_or_update_repo_and_install_requirements "https://github.com/cubiq/ComfyUI_IPAdapter_plus" "custom_nodes/ComfyUI_IPAdapter_plus"
clone_or_update_repo_and_install_requirements "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"  "custom_nodes/ComfyUI-Advanced-ControlNet"  
clone_or_update_repo_and_install_requirements "https://github.com/Acly/comfyui-inpaint-nodes" "custom_nodes/comfyui-inpaint-nodes"
clone_or_update_repo_and_install_requirements "https://github.com/chflame163/ComfyUI_LayerStyle" "custom_nodes/ComfyUI_LayerStyle"  
clone_or_update_repo_and_install_requirements "https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92" "custom_nodes/ComfyUI-QualityOfLifeSuit_Omar92"
clone_or_update_repo_and_install_requirements "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes" "custom_nodes/Derfuu_ComfyUI_ModdedNodes"
clone_or_update_repo_and_install_requirements "https://github.com/EllangoK/ComfyUI-Post-Processing-nodes" "custom_nodes/ComfyUI-Post-Processing-nodes"
clone_or_update_repo_and_install_requirements "https://github.com/jags111/ComfyUI_Jags_VectorMagic" "custom_nodes/ComfyUI_Jags_VectorMagic"
clone_or_update_repo_and_install_requirements "https://github.com/melMass/comfy_mtb"   "custom_nodes/comfy_mtb"  
clone_or_update_repo_and_install_requirements "https://github.com/AuroBit/ComfyUI-OOTDiffusion"   "custom_nodes/ComfyUI-OOTDiffusion"  
clone_or_update_repo_and_install_requirements "https://github.com/kijai/ComfyUI-KJNodes"  "custom_nodes/ComfyUI-KJNodes"  
clone_or_update_repo_and_install_requirements "https://github.com/kijai/ComfyUI-SUPIR"  "custom_nodes/ComfyUI-SUPIR" 
clone_or_update_repo_and_install_requirements "https://github.com/kijai/ComfyUI-depth-fm"  "custom_nodes/ComfyUI-depth-fm" 
clone_or_update_repo_and_install_requirements "https://github.com/viperyl/ComfyUI-BiRefNet" "custom_nodes/ComfyUI-BiRefNet" 
clone_or_update_repo_and_install_requirements "https://github.com/gokayfem/ComfyUI-Texture-Simple" "custom_nodes/ComfyUI-Texture-Simple"
clone_or_update_repo_and_install_requirements "https://github.com/ZHO-ZHO-ZHO/ComfyUI-APISR" "custom_nodes/ComfyUI-APISR"




echo "Downloading AnimateDiff Models..."  

download_hf_model  "https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.f16.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/hotshotco/SDXL-512/resolve/main/hsxl_base_1.0.f16.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_scribble.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models" 
download_hf_model  "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_8step_comfyui.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models"  
download_hf_model  "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_4step_comfyui.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models"
download_hf_model  "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_2step_comfyui.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models"
download_hf_model  "https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_1step_comfyui.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/models"

echo "Downloading Vae..."  


download_hf_model  https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-emaruned.safetensors  "models/vae" 
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/taesdxl.safetensors  "models/vae" 
download_hf_model  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors  "models/vae" 

echo "Downloading Controlnet..."  

download_hf_model  https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors  "models/controlnet" 
download_hf_model  https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors  "models/controlnet" 
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet-sdxl-canny-mid.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet-sdxl-depth-mid.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet_scribble_sd15.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15s2_lineart_anime.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15_lineart.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15_canny_fp16.safetensors  "models/controlnet"
download_hf_model  https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet_depth_sd15.safetensors  "models/controlnet"


download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime_beta.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime_v2.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors  "custom_nodes/ControlNet-LLLite-ComfyUI/models"
download_hf_model  https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors  "models/controlnet"

echo "Downloading LLavacheckpoints..."  

download_hf_model  https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q5_K_M.gguf  "models/LLavacheckpoints" 
download_hf_model  https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf  "models/LLavacheckpoints" 
download_hf_model  https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf  "models/LLavacheckpoints" 
download_hf_model  https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q3_K_XS.gguf  "models/LLavacheckpoints" 
download_hf_model  https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q3_K_M.gguf  "models/LLavacheckpoints" 

echo "Downloading IPAdapter Plus..."  
download_hf_model  https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapterlus-face_sdxl_vit-h.safetensors  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapterlus_sdxl_vit-h.safetensors"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapterlus-face_sd15.safetensors"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapterlus_sd15.safetensors"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceidlusv2_sd15.bin"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/ip-adapter_sdxl.safetensors"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 
download_hf_model  "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors"  "custom_nodes/ComfyUI_IPAdapter_plus/models" 

echo "Downloading ClipVision..."  

download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"  "models/clip_vision" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/sd15_model.safetensors"  "models/clip_vision" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/sdxl_model.safetensors"  "models/clip_vision" 
download_hf_model  "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceidlusv2_sd15.bin"  "models/clip_vision" 

echo "Downloading Upscaler..."  

download_hf_model  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"  "models/upscale_models" 
download_hf_model  "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"  "models/upscale_models" 
download_hf_model  "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"  "models/upscale_models" 

echo "Downloading Lora..."  

download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/lcm-lora-sdv1-5.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/lcm-lora-sdxl.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/3DMM_V12_sd15.safetensors"  "models/loras"  
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/add_detail.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/anime_lineart_lora.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/epi_noiseoffset2_sd15.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/game_bottle_lora.safetensors"  "models/loras" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/game_sword_lora.safetensors"  "models/loras" 

echo "Downloading Motion Lora..."  

download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt"   "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/StopMotionAnimation.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 
download_hf_model  "https://huggingface.co/ArtGAN/Controlnet/resolve/main/shatterAnimatediff_v10.safetensors"  "custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora" 

echo "Downloading SUPIR..."  

download_hf_model  "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt"  "models/checkpoints" 
download_hf_model  "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0F.ckpt"  "models/checkpoints" 

echo "Downloading BiRefNet..."  
clone_or_update_repo_and_install_requirements "https://huggingface.co/ViperYX/BiRefNet" "models/BiRefNet"




echo "Starting the server..."


python main.py --listen "0.0.0.0"