import subprocess

failures = []

with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        print(f"\n🔧 Installing: {line}")
        try:
            subprocess.check_call(["pip", "install", line, "-i", "https://pypi.org/simple"])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {line}")
            failures.append(line)

print("\n✅ Installation finished.")
if failures:
    print("❌ The following packages failed to install:")
    for item in failures:
        print("  -", item)
else:
    print("🎉 All packages installed successfully.")
