# Virtual mic on Windows (VB-CABLE)

1. Download VB-CABLE from the vendor: <https://vb-audio.com/Cable/>.
2. Extract it, right-click `VBCABLE_Setup_x64.exe` and choose **Run as administrator**.
3. Reboot.
4. Check the devices are present:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\detect_virtual_mic_windows.ps1
   ```

5. In the SilentDictate config set `output_mode = "virtual_mic"`. `virtual_mic_device` already defaults to `CABLE Input`.
6. In the receiving app, select **CABLE Output** as the microphone. SilentDictate plays into "CABLE Input", and other apps listen on "CABLE Output".
7. Apps that only use the default device need **CABLE Output** set as the default in Settings > System > Sound > Input.

Run `silentdictate doctor` to confirm.
