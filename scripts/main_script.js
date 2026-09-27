function $(id)
{
    return document.getElementById(id);
}

const cloudDescriptions = new Map()
// V V ENTER API URL FROM PYTHON TERMINAL HERE V V
const apiUrl = "http://127.0.0.1:8000"

cloudDescriptions.set("cirrus", 
    "Cirrus clouds are high-altitude clouds that are composed primarily of ice crystals. They are easily identifiable by their distinct wispy,\
    thin texture. They can take a variety of forms including comma-shaped wisps, fibrous clumps (spissatus), and can blanket the entire sky in a \
    thin and transparent layer of ice crystals. Their icy composition allows many optical phenomena to be seen in cirrus-laden skies, including sun-rays\
    and iridescence. Some cirrus clouds are formed by human activities like plane contrails. Cirrus clouds hint at fair weather.");

cloudDescriptions.set("cumulus", "Cumulus clouds are the classic cloud type that people think of when they imagine clouds. They are low-level clouds\
    that take a puffy heaped shape. They are capable of taking many shapes with some cumulus clouds claiming a towering shape while others taking a wider \
    shape. With the right weather conditions, they can grow and tower upwards higher into the atmosphere to form clouds associated with rain and storms.\
    However, they are signs of fair and fine weather conditions.")

cloudDescriptions.set("cumulonimbus", "Cumulonimbus clouds are magnificient lofty clouds that most often evolve into towering storm clouds. They start as cumulus \
    clouds and develop upwards into atmosphere until they reach a ceiling where water droplets freeze into ice crystals, giving cumulonimbus clouds their characteristic anvil\
    shape. Beneath these clouds, falling pockets of air sometimes give the cloud base a texture that resemble the udders of the cow. Cumulonimbus clouds can bring thunder and lightning\
    as well as heavy rainfall so shelter should be sought.")

cloudDescriptions.set("stratus", "Stratus clouds are featureless clouds that often blanket the entire sky. They are grey to dark grey in colour and form low in the atmosphere. They can\
    fall low to the ground, which results in fog. When stratus clouds bring rainfall, they are termed nimbostratus clouds. They can be opaque or transparent, which for the former often means\
    dull and dreary days. Some stratus clouds can have an undulating appearance at their base. They often bring just dull, foggy and misty weather.")

async function uploadImage()
{
    if ($("cloud_image_file").files.length == 0)
    {
        alert("Please upload an image to classify.");
        return;
    }
    const imageFile = $("cloud_image_file").files[0];
    const formData = new FormData();
    formData.append("file", imageFile);

    try
    {
        const response = await fetch(apiUrl +"/predict", 
            {
                method: "POST",
                body: formData
            });

        const data = await response.json();

        if (!response.ok)
        {
            alert("Error: "+data.detail);
            return;
        }

        $("predicted_cloud").innerText = data.predicted_class.toLowerCase();
        $("prediction_confidence").innerText = parseFloat(data.confidence).toFixed(2);
        console.log(data.predicted_class)
        $("cloud_description").innerText = cloudDescriptions.get(data.predicted_class.toLowerCase())

        let predictionsTable = $("predictions_table")
        predictionsTable.innerHTML=`
            <tr>
                <th>Classification</th>
                <th>Confidence</th>
            </tr>`

        for (const [cloudType, prediction_score] of Object.entries(data.predictions))
        {
            predictionsTable.innerHTML += `
            <tr>
                <td>${cloudType}</td>
                <td>${prediction_score}%</td>
            </tr>`
        }

        $("result_summary").style.display = "block";

    }
    catch (error)
    {
        console.error("Error uploading image: ", error);
        alert("Failed to connect to the backend model server");
    }


}

function previewImage()
{
    const inputFile = $("cloud_image_file");
    const previewImage = $("image_preview")

    if (inputFile.files.length > 0)
    {
        const file = inputFile.files[0];

        const fileReader = new FileReader()

        fileReader.onload = function(e){
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
        }

        fileReader.readAsDataURL(file);
    }
    else
    {
        previewImage.style.display = "none";
    }
}