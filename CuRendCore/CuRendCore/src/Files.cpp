#include "Files.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fstream>

namespace CRC
{

PngFile::~PngFile()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    Unload();
}

HRESULT PngFile::Load()
{
    stbi_info(rattr.path.c_str(), &width, &height, &channels);
    stbi_uc* pixels = stbi_load(rattr.path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) return E_FAIL;

    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, rattr.path);

    data = new DWORD[width * height];

    int pixelIndex = 0;
    for(int y = 0; y <= height; y++)
    {
        for(int x = 0; x <= width; x++)
        {
            if (x < width && y < height)
            {
                data[x+y*width] = (pixels[pixelIndex * 4 + 3] << 24) | 
                                (pixels[pixelIndex * 4] << 16) | 
                                (pixels[pixelIndex * 4 + 1] << 8) | 
                                pixels[pixelIndex * 4 + 2];
                pixelIndex += 1;
            }
        }
    }

    stbi_image_free(pixels);

    return S_OK;
}

HRESULT PngFile::Unload()
{
    if (data != nullptr)
    {
        CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, rattr.path);
        delete[] data;
        data = nullptr;
    }
    return S_OK;
}

ObjFile::~ObjFile()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    Unload();
}

HRESULT ObjFile::Load()
{
    std::ifstream file(rattr.path);
    if (file.fail()) return E_FAIL;

    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, rattr.path);

    Unload();

    boundingBox = new BoundingBox();
    boundingBox->StartToCreate();
    std::string line;
    while (std::getline(file, line)) 
    {
        int space1 = line.find_first_of(" ");
        std::string type = line.substr(0, space1);
        std::string contents = line.substr(space1 + 1, line.size() - space1);

        if (type == "#") continue;
        else if (type == "") continue;
        else if (type == "mtllib") continue;
        else if (type == "g") continue;
        else if (type == "usemtl") continue;
        else if (type == "v")
        {
            Vec3d vec
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")))
            );

            wv.push_back(vec);
            boundingBox->AddPoint(vec);
        }
        else if (type == "vt")
        {
            Vec2d vec
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.size() - contents.find_first_of(" ")))
            );

            uv.push_back(vec);
        }
        else if (type == "vn")
        {
            Vec3d vec
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")))
            );

            normal.push_back(vec);
        }
        else if (type == "f")
        {
            std::vector<std::string> innerContents;
            innerContents.push_back(contents.substr(0, contents.find_first_of(" ")));
            innerContents.push_back(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" ")));
            innerContents.push_back(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")));

            Polygon polygon;
            polygon.StartToCreate();

            for (int i = 0; i < innerContents.size(); i++)
            {
                std::string innerContent = innerContents[i];
                
                int vNum = std::stoi(innerContent.substr(0, innerContent.find_first_of("/"))) - 1;
                int uvNum 
                = std::stoi
                (
                    innerContent.substr(innerContent.find_first_of("/") + 1, 
                    innerContent.find_last_of("/") - innerContent.find_first_of("/"))
                ) - 1;

                polygon.AddIndex(vNum, uvNum);
            }

            int vnNum = std::stoi(innerContents[0].substr(innerContents[0].find_last_of("/") + 1, innerContents[0].size() - innerContents[0].find_last_of("/")));
            polygon.SetNormal(normal[vnNum - 1]);

            polygons.push_back(polygon);
        }
    }

    boundingBox->Update();
}

HRESULT ObjFile::Unload()
{   
    wv.clear();
    uv.clear();
    normal.clear();

    if (boundingBox != nullptr)
    {
        CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, rattr.path);
        
        delete boundingBox;
        boundingBox = nullptr;
    }
    polygons.clear();

    return S_OK;
}

} // namespace CRC
