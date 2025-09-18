import discord
from discord.ext import commands
import os
import io
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Configure Discord intents and bot prefix
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Load class names from labels.txt
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Load your trained ResNet model
num_classes = len(class_names)

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transforms matching training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Energy usage dictionary
usage_levels = {
    "laptop": "Low usage",
    "air conditioner": "High usage",
    "washing machine": "Medium usage",
    "microwave": "Low usage",
    "tv": "Medium usage",
    "smartphone": "Low usage",
    "tablet": "Low usage",
    "camera": "Low usage",
    "headphones": "Low usage",
    "speaker": "Low usage",
    "game console": "Medium usage",
    "router": "Low usage",
    "printer": "Low usage"
}

# Electronics info dictionary with pros, cons, eco tips
electronics_info = {
    "laptop": {
        "pros": "Portable, versatile, energy-efficient compared to desktops.",
        "cons": "Battery degradation over time, manufacturing impact.",
        "eco_tip": "Extend lifespan by upgrading RAM/SSD and recycling old devices responsibly."
    },
    "air conditioner": {
        "pros": "Effective cooling, improves indoor comfort.",
        "cons": "High energy consumption, greenhouse gas emissions.",
        "eco_tip": "Use energy-efficient models, maintain filters, and set temperature wisely."
    },
    "washing machine": {
        "pros": "Saves time and water compared to hand washing.",
        "cons": "Consumes electricity and water, may have environmental impact if inefficient.",
        "eco_tip": "Use full loads, select eco modes, and maintain regularly."
    },
    "microwave": {
        "pros": "Fast cooking, energy-efficient for small portions.",
        "cons": "Limited cooking capability compared to ovens.",
        "eco_tip": "Use microwave for reheating to save energy."
    },
    "tv": {
        "pros": "Entertainment and information source.",
        "cons": "Energy consumption varies, screen manufacturing impacts environment.",
        "eco_tip": "Turn off when not in use and choose energy-efficient displays."
    },
    "smartphone": {
        "pros": "Portable, versatile communication and computing device.",
        "cons": "Short lifespan, resource-intensive manufacturing.",
        "eco_tip": "Use cases and screen protectors to prolong lifespan, recycle old phones."
    },
    "tablet": {
        "pros": "Portable and convenient for media consumption and light tasks.",
        "cons": "Battery life degrades, resource use in production.",
        "eco_tip": "Avoid overcharging and recycle responsibly."
    },
    "camera": {
        "pros": "High-quality image capture, digital convenience.",
        "cons": "Battery and accessory waste, manufacturing footprint.",
        "eco_tip": "Use rechargeable batteries and recycle old electronics."
    },
    "headphones": {
        "pros": "Personal audio, portable.",
        "cons": "Short lifespan for cheaper models.",
        "eco_tip": "Choose durable brands and recycle properly."
    },
    "speaker": {
        "pros": "Portable audio output.",
        "cons": "Battery waste, plastic use.",
        "eco_tip": "Use wired speakers when possible to reduce battery waste."
    },
    "game console": {
        "pros": "Entertainment and social connection.",
        "cons": "High power consumption during use and standby.",
        "eco_tip": "Turn off fully when not in use, unplug accessories."
    },
    "router": {
        "pros": "Provides internet connectivity.",
        "cons": "Consumes electricity continuously.",
        "eco_tip": "Use energy-efficient routers and restart periodically."
    },
    "printer": {
        "pros": "Convenient document output.",
        "cons": "Ink and paper waste, energy use.",
        "eco_tip": "Print only when necessary, use duplex mode."
    }
}

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.command()
async def classify(ctx):
    if not ctx.message.attachments:
        await ctx.send("Please attach an image for classification.")
        return

    attachment = ctx.message.attachments[0]
    image_bytes = await attachment.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        await ctx.send("Error opening image. Please try again with a valid image file.")
        return

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence_score, predicted_idx = torch.max(probabilities, 0)

    class_name = class_names[predicted_idx.item()]
    confidence_score = confidence_score.item()
    usage = usage_levels.get(class_name.lower(), "Unknown")

    await ctx.send(
        f"🔍 **Prediction:** **{class_name}**\n"
        f"📊 **Confidence:** **{confidence_score:.2%}**\n"
        f"⚡ **Estimated Energy Usage:** **{usage}**"
    )

@bot.command()
async def info(ctx, *, device: str):
    device = device.lower()
    if device in electronics_info:
        info = electronics_info[device]
        response = (
            f"**{device.title()} Info:**\n"
            f"Pros: {info['pros']}\n"
            f"Cons: {info['cons']}\n"
            f"Eco Tip: {info['eco_tip']}"
        )
    else:
        response = "Sorry, I don't have info on that device."

    await ctx.send(response)

@bot.command()
async def howtouse(ctx, *, device: str):
    device = device.lower()
    if device in electronics_info:
        eco_tip = electronics_info[device]['eco_tip']
        response = f"**Sustainable Use Tip for {device.title()}:** {eco_tip}"
    else:
        response = "Sorry, I don't have usage tips for that device."

    await ctx.send(response)

@bot.command()
async def helpme(ctx):
    help_text = (
        "**🌿 Available Commands:**\n"
        "`!classify` - 📸 Attach an image of an electronic device to classify it and get its energy usage level.\n"
        "`!info <device>` - ℹ️ Get detailed info about a specific electronic device (e.g., 💻 laptop, ❄️ air conditioner).\n"
        "`!howtouse <device>` - ♻️ Get tips on how to use a specific device sustainably.\n"
        "`!ping` - 🛰️ Check if the bot is responsive.\n"
        "`!about` - 📖 Learn more about this bot.\n"
        "`!source` - 💻 Get the link to the bot’s source code.\n"
        "`!invite` - 🔗 Get an invite link to add the bot to your server.\n"
        "`!support` - 🆘 Join the support server for help and updates.\n"
        "\n**🌟 How to Use:**\n"
        "⚡ Replace `<device>` with the name of the electronic device you want information about "
        "(e.g., laptop, air conditioner, washing machine, etc.).\n"
        "💡 Make sure to attach an image when using the `!classify` command.\n"
        "📷 For best results, use clear images with good lighting.\n"
        "🤝 Feel free to reach out if you have any questions or need assistance!\n"
        "🌍 Happy to help you make more eco-friendly choices! :earth_africa:\n"
        "🔁 Use `!helpme` to see this message again. :smiley:\n"
        "🌱 Stay green and sustainable! :seedling:\n"
    )
    await ctx.send(help_text)

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command()
async def about(ctx):
    about_text = (
        "I am EcoFriendlyBot, designed to help you classify electronic devices and provide information on their energy usage and sustainable practices. "
        "Use `!helpme` to see what I can do!"
    )
    await ctx.send(about_text)

@bot.command()
async def source(ctx):
    source_text = (
        "You can find my source code on GitHub: https://github.com/Gevonda108/Eco-Friendly-Bot"
    )
    await ctx.send(source_text)

@bot.command()
async def invite(ctx):
    invite_text = (
        "Invite me to your server using this link: https://discord.com/oauth2/authorize?client_id=1415680685858881617&permissions=2048&integration_type=0&scope=bot"
    )
    await ctx.send(invite_text)

@bot.command()
async def support(ctx):
    support_text = (
        "Join the support server for help and updates: https://discord.gg/4ZfHhYk3"
    )
    await ctx.send(support_text)

# Run the bot with the token loaded from .env
bot.run(TOKEN)
