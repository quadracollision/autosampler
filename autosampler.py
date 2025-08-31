import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import random
import os
import time
import datetime
import numpy as np
import traceback

DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_BPM = 120
OUTPUT_DIR = "generated_batch"
# =====================================================================================

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_faq():
    clear_console()
    print("\n" + "="*60)
    print("--- FAQ: Quadacollision AutoSample v0.2 ---")
    print("* Updated August 31, 2025")
    print("* By Jacob Pereira ")
    print("="*60)
    print("\nTEMPERATURE:")
    print("* Controls randomness and creativity in generation")
    print("* Lower values (0.1-0.7): More predictable, consistent results")
    print("* Higher values (1.0-1.5): More creative, varied, experimental")
    print("\nGUIDANCE SCALE:")
    print("* Controls how closely the AI follows your text prompt")
    print("* Lower values (1.0-3.0): More creative freedom, looser interpretation")
    print("* Higher values (5.0-10.0): Stricter adherence to prompt")
    print("\nAbout:")
    print("\n" + "="*60)
    input("\nPress Enter to return to main menu...")

def show_settings():
    global DEFAULT_BPM, DEFAULT_TEMPERATURE, DEFAULT_GUIDANCE_SCALE
    
    while True:
        clear_console()
        print("\n" + "="*50)
        print("--- SETTINGS ---")
        print("="*50)
        print(f"\nCurrent Default Values:")
        print(f"  [1] BPM: {DEFAULT_BPM}")
        print(f"  [2] Temperature: {DEFAULT_TEMPERATURE}")
        print(f"  [3] Guidance Scale: {DEFAULT_GUIDANCE_SCALE}")
        print(f"  [0] Back to Main Menu")
        
        try:
            selection = input("\nSelect setting to modify (0-3): ")
            selected_index = int(selection)
            
            if selected_index == 0:
                break
            elif selected_index == 1:
                while True:
                    try:
                        new_bpm = input(f"\nEnter new default BPM (current: {DEFAULT_BPM}): ").strip()
                        if new_bpm == "":
                            break
                        new_bpm_val = int(new_bpm)
                        if new_bpm_val <= 0:
                            print("BPM must be positive. Please try again.")
                            continue
                        DEFAULT_BPM = new_bpm_val
                        print(f"Default BPM updated to {DEFAULT_BPM}")
                        input("Press Enter to continue...")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except EOFError:
                        print("\nInput interrupted. Returning to settings...")
                        break
            elif selected_index == 2:
                while True:
                    try:
                        new_temp = input(f"\nEnter new default Temperature (current: {DEFAULT_TEMPERATURE}): ").strip()
                        if new_temp == "":
                            break
                        new_temp_val = float(new_temp)
                        if new_temp_val <= 0:
                            print("Temperature must be positive. Please try again.")
                            continue
                        DEFAULT_TEMPERATURE = new_temp_val
                        print(f"Default Temperature updated to {DEFAULT_TEMPERATURE}")
                        input("Press Enter to continue...")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except EOFError:
                        print("\nInput interrupted. Returning to settings...")
                        break
            elif selected_index == 3:
                while True:
                    try:
                        new_guidance = input(f"\nEnter new default Guidance Scale (current: {DEFAULT_GUIDANCE_SCALE}): ").strip()
                        if new_guidance == "":
                            break
                        new_guidance_val = float(new_guidance)
                        if new_guidance_val <= 0:
                            print("Guidance Scale must be positive. Please try again.")
                            continue
                        DEFAULT_GUIDANCE_SCALE = new_guidance_val
                        print(f"Default Guidance Scale updated to {DEFAULT_GUIDANCE_SCALE}")
                        input("Press Enter to continue...")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except EOFError:
                        print("\nInput interrupted. Returning to settings...")
                        break
            else:
                print("Invalid number. Please try again.")
                input("Press Enter to continue...")
        except ValueError:
            print("Invalid input. Please enter a number.")
            input("Press Enter to continue...")
        except EOFError:
            print("\nInput interrupted. Returning to main menu...")
            break

# --- Main Script ---
if __name__ == "__main__":
    # --- Model Loading (Done once at the start) ---
    print(f"--- Quadracollision AutoSample v0.2 (08/31/25) ---")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}")

    try:
        print("Loading MusicGen model (this may take a moment)...")
        # Set cache directory to local folder
        cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small", cache_dir=cache_dir)
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torch_dtype=torch_dtype, cache_dir=cache_dir).to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        exit()

    # --- Main Menu Loop ---
    while True:
        try:
            clear_console()
            print("\n" + "="*50)
            print("Quadracollision AutoSample v0.2 (August 31, 2025)")
            print("="*50)
            print("\nOptions:")
            print("  [1] Generate Audio")
            print("  [2] FAQ")
            print("  [3] Settings")
            print("  [0] Quit")

            # --- Get User Mode Selection ---
            while True:
                try:
                    selection = input("\nSelect an option (0-3): ")
                    selected_index = int(selection)
                    if selected_index == 0:
                        print("\nExiting. Goodbye!")
                        exit()
                    elif selected_index == 1:
                        break  # Continue to generation setup
                    elif selected_index == 2:
                        show_faq()
                        break  # Return to main menu after FAQ
                    elif selected_index == 3:
                        show_settings()
                        break  # Return to main menu after settings
                    else:
                        print("Invalid number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except EOFError:
                    print("\nInput interrupted. Exiting...")
                    exit()
            
            if selected_index in [2, 3]:  # FAQ or Settings was selected, continue to next iteration
                continue
            
            # --- Set Generation Parameters ---
            clear_console()
            print("\n" + "="*50)
            print("--- GENERATION SETUP ---")
            print("="*50)
            
            # Get prompts (allow multiple)
            prompts = []
            print("\nEnter your prompts (press Enter on empty line to finish):")
            while True:
                try:
                    prompt = input(f"Prompt {len(prompts) + 1}: ").strip()
                    if prompt == "":
                        if len(prompts) == 0:
                            print("You must enter at least one prompt.")
                            continue
                        break
                    prompts.append(prompt)
                except EOFError:
                    print("\nInput interrupted. Returning to main menu...")
                    raise KeyboardInterrupt()
            
            # Combine all prompts
            manual_prompt = ", ".join(prompts)
            print(f"\nCombined prompt: {manual_prompt}")
            
            # Get BPM
            while True:
                try:
                    bpm_input = input(f"\nEnter BPM (default {DEFAULT_BPM}): ").strip()
                    if bpm_input == "":
                        bpm = DEFAULT_BPM
                    else:
                        bpm = int(bpm_input)
                        if bpm <= 0:
                            print("BPM must be positive. Please try again.")
                            continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except EOFError:
                    print("\nInput interrupted. Returning to main menu...")
                    raise KeyboardInterrupt()
            
            # Get Temperature
            while True:
                try:
                    temp_input = input(f"\nEnter Temperature (default {DEFAULT_TEMPERATURE}): ").strip()
                    if temp_input == "":
                        temperature = DEFAULT_TEMPERATURE
                    else:
                        temperature = float(temp_input)
                        if temperature <= 0:
                            print("Temperature must be positive. Please try again.")
                            continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except EOFError:
                    print("\nInput interrupted. Returning to main menu...")
                    raise KeyboardInterrupt()
            
            # Get Guidance Scale
            while True:
                try:
                    guidance_input = input(f"\nEnter Guidance Scale (default {DEFAULT_GUIDANCE_SCALE}): ").strip()
                    if guidance_input == "":
                        guidance_scale = DEFAULT_GUIDANCE_SCALE
                    else:
                        guidance_scale = float(guidance_input)
                        if guidance_scale <= 0:
                            print("Guidance scale must be positive. Please try again.")
                            continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except EOFError:
                    print("\nInput interrupted. Returning to main menu...")
                    raise KeyboardInterrupt()
            
            # Create subfolder name
            sanitized_prompt = "".join(c for c in manual_prompt if c.isalnum() or c in (' ','.','_')).rstrip().strip()
            subfolder_name = f"generated_{sanitized_prompt[:30]}"

            # --- Get User Input for Duration ---
            while True:
                try:
                    duration_input = input("Enter audio duration in seconds (e.g., 10 or 0.5): ")
                    AUDIO_DURATION_SECONDS = float(duration_input)
                    if AUDIO_DURATION_SECONDS > 0:
                        break
                    else:
                        print("Duration must be positive. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except EOFError:
                    print("\nInput interrupted. Returning to main menu...")
                    raise KeyboardInterrupt()
            
            specific_output_dir = os.path.join(OUTPUT_DIR, subfolder_name)
            if not os.path.exists(specific_output_dir):
                os.makedirs(specific_output_dir)

            # --- Generation Loop ---
            clear_console()
            print("\n" + "="*50)
            print("--- AUTOSAMPLING IN PROGRESS ---")
            print("="*50)
            print(f"Prompt: {manual_prompt}")
            print(f"BPM: {bpm}")
            print(f"Temperature: {temperature}")
            print(f"Guidance Scale: {guidance_scale}")
            print(f"Duration: {AUDIO_DURATION_SECONDS} seconds")
            print("\nStarting generation... Press Ctrl+C to stop and return to the main menu.")
            time.sleep(2)
            
            generation_count = 0
            while True:
                generation_count += 1
                prompt = f"{manual_prompt} at {bpm} BPM"
                
                print("\n" + "="*50)
                print(f"Generating audio #{generation_count} for prompt: \"{prompt}\"")
                
                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                max_new_tokens = int(AUDIO_DURATION_SECONDS * 50)
                
                generation_start_time = time.time()
                audio_values = model.generate(
                    **inputs, do_sample=True, max_new_tokens=max_new_tokens, 
                    guidance_scale=guidance_scale, temperature=temperature
                )
                generation_time = time.time() - generation_start_time
                print(f"Audio generated in {generation_time:.2f} seconds.")

                sampling_rate = model.config.audio_encoder.sampling_rate
                sanitized_prompt_for_filename = "".join(c for c in manual_prompt if c.isalnum() or c in (' ','.','_')).rstrip().strip()

                base_filename = f"{bpm}BPM - {sanitized_prompt_for_filename[:50]} ({generation_count}).wav"
                filepath = os.path.join(specific_output_dir, base_filename)
                
                counter = 1
                while os.path.exists(filepath):
                    name, ext = os.path.splitext(base_filename)
                    filepath = os.path.join(specific_output_dir, f"{name} - copy{counter}{ext}")
                    counter += 1
                
                print(f"Saving to: {filepath}")
                audio_data_to_save = audio_values[0, 0].cpu().numpy().astype(np.float32)
                scipy.io.wavfile.write(filepath, rate=sampling_rate, data=audio_data_to_save)
                print(f"--- Generation #{generation_count} Complete ---")

        except KeyboardInterrupt:
            print("\n\nAutoSampling stopped. Returning to main menu...")
            continue
        except Exception as e:
            print(f"\n\nAn unexpected error occurred during this operation: {e}")
            traceback.print_exc()
            print("Returning to main menu. Please wait...")
            time.sleep(2)
            continue
