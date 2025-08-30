# Simple AtomForge Converter Pseudocode

## Overview
Convert natural language material descriptions into AtomForge DSL using AI and Materials Project data.

## Main Algorithm

```
FUNCTION ConvertMaterial(user_input)
    // Step 1: Understand what the user wants
    query = AnalyzeQuery(user_input)
    
    // Step 2: Get material data from database
    material_data = GetMaterialData(query)
    
    // Step 3: Add computational settings
    enriched_data = AddComputationalSettings(material_data)
    
    // Step 4: Generate AtomForge DSL
    dsl = GenerateDSL(material_data, enriched_data)
    
    // Step 5: Check if DSL is correct
    final_dsl = ValidateAndFix(dsl)
    
    RETURN final_dsl
END FUNCTION
```

## Step 1: Query Analysis

```
FUNCTION AnalyzeQuery(user_input)
    IF AI_available THEN
        // Use AI to understand complex descriptions
        result = AskAI("What material is this: " + user_input)
        RETURN ParseAIResponse(result)
    ELSE
        // Simple keyword extraction
        IF user_input contains "mp-" THEN
            RETURN user_input  // Direct Materials Project ID
        ELSE
            elements = ExtractChemicalElements(user_input)
            RETURN elements
        END IF
    END IF
END FUNCTION
```

## Step 2: Material Data Retrieval

```
FUNCTION GetMaterialData(query)
    // Connect to Materials Project / COD database
    database = ConnectToDb()
    
    // Search by elements
    material = database.SearchByElements(query)
        
    IF material not found THEN
        // Try chemical system search
        material = database.SearchByChemicalSystem(query)
    END IF
    
    IF material not found THEN ERROR
    END IF
    
    RETURN material
END FUNCTION
```

## Step 3: DSL Enrichment

```
FUNCTION AddComputationalSettings(material_data)
    IF AI_available THEN
        // Ask AI for appropriate computational settings 
        settings = AskAI("What computational settings for " + material_data.formula)
        RETURN ParseAIResponse(settings)
    ELSE
        RETURN DEFAULT
    END IF
END FUNCTION
```

## Step 3: DSL Generation

```
FUNCTION GenAtomForgeDSL(material_data)
    IF AI_available THEN
        prompt = "Generate AtomForge DSL for: " + material_data 
        atomforge_dsl = AskAI(prompt)
        RETURN CleanDSL(atomforge_dsl)
    ELSE RETURN ConvertDSL(material_data) // manual conversion
    END IF
END FUNCTION
```

## Step 4: Validation and Feedback loop

```
FUNCTION AtomForgeValidator(dsl)
    // Try to parse the DSL
    TRY
        ParseAtomForgeDSL(dsl)
        RETURN atomforge_dsl  // DSL is valid
    CATCH error
        IF AI_available THEN
            // Ask AI to fix the DSL
            fixed_dsl = AskAI("Fix this DSL: " + dsl + " Error: " + error)
            RETURN fixed_dsl
        ELSE ERROR "DSL is invalid"
        END IF
    END TRY
END FUNCTION
```

## Helper Functions

```
FUNCTION ExtractChemicalElements(text)
    // Find chemical element symbols in text
    elements = []
    FOR each word in text
        IF word matches chemical element pattern THEN
            ADD word to elements
        END IF
    END FOR
    RETURN elements
END FUNCTION

FUNCTION CleanDSL(dsl)
    // Remove markdown formatting
    dsl = RemoveMarkdown(dsl)
    // Fix common syntax errors
    dsl = FixSyntaxErrors(dsl)
    RETURN dsl
END FUNCTION

FUNCTION AskAI(prompt)
    // Send prompt to AI service
    response = AI_Service.Send(prompt)
    RETURN response
END FUNCTION
```

## Data Structures

```
STRUCTURE MaterialQuery
    query: string          // What to search for
    description: string    // User's original input
    confidence: number     // How sure we are (0-1)
END STRUCTURE

STRUCTURE MaterialData
    id: string            // Materials Project ID
    formula: string       // Chemical formula
    structure: object     // Crystal structure
    properties: object    // Material properties
END STRUCTURE
```


