from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import os

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from llmcore import LLM, LLMConfig, LLMChainBuilder, PromptTemplate, CodebaseEmbeddings, Embeddings
from llmcore.tools.knowledge_graph import KnowledgeGraph, Concept, ConceptCategory
from llmcore.memory import MemoryManager

console = Console()

class AIFinancialAnalysisSystem:
    def __init__(self):
        console.print(Panel("AI-Powered Financial Analysis System 📊", title="Initializing"))
        
        self.llm_analysis = LLM(provider="anthropic", model="claude-3-5-sonnet-20240620", config=LLMConfig(temperature=0.4, max_tokens=3000))
        self.llm_summary = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.6, max_tokens=2000))
        self.llm_relationship = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.5, max_tokens=1500))
        
        self.knowledge_graph = KnowledgeGraph()
        
        console.print("✨ Initializing knowledge graph and LLM chains")
        
        self.stock_analysis_chain = self.create_stock_analysis_chain()
        self.news_analysis_chain = self.create_news_analysis_chain()
        self.market_insight_chain = self.create_market_insight_chain()
    
    def create_stock_analysis_chain(self):
        price_analysis_template = PromptTemplate(
            "Analyze the following stock price data for {{symbol}}:\n\n{{price_data}}\n\n"
            "Provide a summary of price trends and key statistics.",
            required_params={"symbol": str, "price_data": str},
            output_json_structure={"price_summary": str, "key_stats": Dict[str, float]}
        )
        
        fundamental_analysis_template = PromptTemplate(
            "Analyze the following company fundamentals for {{symbol}}:\n\n{{company_overview}}\n\n"
            "Provide an analysis of the company's financial health and growth prospects.\n\n"
            "Price Analysis: {{price_analysis.price_summary}}\n"
            "Key Statistics: {{price_analysis.key_stats}}",
            required_params={"symbol": str, "company_overview": str, "price_analysis": Dict[str, Any]},
            output_json_structure={"financial_health": str, "growth_prospects": str}
        )
        
        overall_analysis_template = PromptTemplate(
            "Combine the price analysis and fundamental analysis to provide an overall stock analysis for {{symbol}}:\n\n"
            "Price Analysis: {{price_analysis.price_summary}}\n"
            "Key Statistics: {{price_analysis.key_stats}}\n"
            "Financial Health: {{fundamental_analysis.financial_health}}\n"
            "Growth Prospects: {{fundamental_analysis.growth_prospects}}\n\n"
            "Provide a comprehensive analysis and investment recommendation.",
            required_params={"symbol": str, "price_analysis": Dict[str, Any], "fundamental_analysis": Dict[str, str]},
            output_json_structure={"analysis": str, "recommendation": str, "risk_level": str}
        )
        
        return (
            LLMChainBuilder(self.llm_analysis)
            .add_step(
                template=price_analysis_template.template,
                output_key="price_analysis",
                required_params=price_analysis_template.required_params,
                output_json_structure=price_analysis_template.output_json_structure
            )
            .add_step(
                template=fundamental_analysis_template.template,
                output_key="fundamental_analysis",
                required_params=fundamental_analysis_template.required_params,
                output_json_structure=fundamental_analysis_template.output_json_structure
            )
            .add_step(
                template=overall_analysis_template.template,
                output_key="overall_analysis",
                required_params=overall_analysis_template.required_params,
                output_json_structure=overall_analysis_template.output_json_structure
            )
            .build()
        )
    
    def create_news_analysis_chain(self):
        article_summary_template = PromptTemplate(
            "Summarize the following financial news article:\n\n{{article}}\n\n"
            "Provide a concise summary and extract key points.",
            required_params={"article": Dict[str, Any]},
            output_json_structure={"summary": str, "key_points": List[str]}
        )
        
        news_impact_template = PromptTemplate(
            "Analyze the potential impact of the following news summary on the stock market:\n\n{{article_summary.summary}}\n\n"
            "Key Points:\n{{article_summary.key_points}}\n\n"
            "Provide an analysis of the potential market impact.",
            required_params={"article_summary": Dict[str, Any]},
            output_json_structure={"market_impact": str, "affected_sectors": List[str]}
        )
        
        return (
            LLMChainBuilder(self.llm_summary)
            .add_step(
                template=article_summary_template.template,
                output_key="article_summary",
                required_params=article_summary_template.required_params,
                output_json_structure=article_summary_template.output_json_structure
            )
            .add_step(
                template=news_impact_template.template,
                output_key="news_impact",
                required_params=news_impact_template.required_params,
                output_json_structure=news_impact_template.output_json_structure
            )
            .build()
        )
    
    def create_market_insight_chain(self):
        trend_analysis_template = PromptTemplate(
            "Analyze the following market data and news impacts:\n\n{{market_data}}\n\n{{news_impacts}}\n\n"
            "Identify key market trends and potential opportunities.",
            required_params={"market_data": str, "news_impacts": str},
            output_json_structure={"trends": List[str], "opportunities": List[str]}
        )
        
        strategy_recommendation_template = PromptTemplate(
            "Based on the identified trends and opportunities:\n\n"
            "Trends: {{trends}}\n"
            "Opportunities: {{opportunities}}\n\n"
            "Provide strategic investment recommendations and risk assessments.",
            required_params={"trends": List[str], "opportunities": List[str]},
            output_json_structure={"recommendations": List[str], "risk_assessment": str}
        )
        
        return (
            LLMChainBuilder(self.llm_analysis)
            .add_step(
                template=trend_analysis_template.template,
                output_key="trend_analysis",
                required_params=trend_analysis_template.required_params,
                output_json_structure=trend_analysis_template.output_json_structure
            )
            .add_step(
                template=strategy_recommendation_template.template,
                output_key="strategy_recommendation",
                required_params=strategy_recommendation_template.required_params,
                output_json_structure=strategy_recommendation_template.output_json_structure
            )
            .build()
        )
    
    async def fetch_stock_data(self, symbol: str, api_key: Optional[str]) -> Dict[str, Any]:
        api_key = api_key if api_key is not None else os.getenv("ALPHA_VANTAGE_API_KEY")
        async with aiohttp.ClientSession() as session:
            # Fetch daily stock data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
            async with session.get(url) as response:
                daily_data = await response.json()
            
            # Fetch company overview
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
            async with session.get(url) as response:
                overview = await response.json()
        
        return {"daily_data": daily_data, "overview": overview}

    async def fetch_article_content(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Extract the main content (this might need adjustment based on the website structure)
                content = soup.find('article') or soup.find('main') or soup.find('body')
                return content.get_text() if content else "Failed to extract content"

    async def fetch_news_articles(self, query: str, api_key: Optional[str]) -> List[Dict[str, str]]:
        api_key = api_key if api_key is not None else os.getenv("NEWSAPI_KEY")
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "domains": "finance.yahoo.com,bloomberg.com,reuters.com,cnbc.com,wsj.com",
                "from": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            }
            url = "https://newsapi.org/v2/everything"
            async with session.get(url, params=params) as response:
                news_data = await response.json()
        
        return news_data.get("articles", [])

    async def analyze_stock(self, symbol: str, price_data: str, company_overview: str, relevant_news: List[Dict[str, Any]]):
        console.print(f"📊 Analyzing stock: {symbol}")
        
        # Prepare relevant news summaries
        news_summaries = []
        for news in relevant_news:
            summary = await self.news_analysis_chain.execute_async({
                "article": {
                    "title": news['title'],
                    "description": news['description'],
                    "content": await self.fetch_article_content(news['url']),
                    "url": news['url'],
                    "publishedAt": news['publishedAt']
                }
            })
            news_summaries.append(summary["article_summary"]["summary"])
        
        # Update the stock analysis chain to include news summaries
        result = await self.stock_analysis_chain.execute_async({
            "symbol": symbol,
            "price_data": price_data,
            "company_overview": company_overview,
            "relevant_news": "\n".join(news_summaries)
        })
        
        analysis = result["overall_analysis"]
        await self.knowledge_graph.add_concept(
            name=f"Stock Analysis: {symbol}",
            description=analysis["analysis"],
            category="Stock Analysis",
            confidence=0.85,
            sources=[{"url": "AI-generated", "credibility": 0.8}],
            abstraction_level="expert"
        )
        
        return analysis
    
    async def analyze_news(self, articles: List[Dict[str, Any]]):
        console.print("📰 Analyzing financial news")
        news_impacts = []
        for article in articles:
            console.print(f"Analyzing article: {article['title']}")
            full_content = await self.fetch_article_content(article['url'])
            
            result = await self.news_analysis_chain.execute_async({
                "article": {
                    "title": article['title'],
                    "description": article['description'],
                    "content": full_content,
                    "url": article['url'],
                    "publishedAt": article['publishedAt']
                }
            })
            news_impacts.append(result["news_impact"])
            
            await self.knowledge_graph.add_concept(
                name=f"News Impact: {article['title'][:50]}...",
                description=result["news_impact"]["market_impact"],
                category="Financial News",
                confidence=0.9,
                sources=[{"url": article['url'], "credibility": 0.85}],
                abstraction_level="intermediate"
            )
        
        return news_impacts
    
    async def generate_market_insights(self, market_data: str, news_impacts: List[Dict[str, Any]]):
        console.print("💡 Generating market insights")
        result = await self.market_insight_chain.execute_async({
            "market_data": market_data,
            "news_impacts": str(news_impacts)
        })
        
        insights = result["strategy_recommendation"]
        await self.knowledge_graph.add_concept(
            name="Market Insights",
            description=str(insights["recommendations"]),
            category="Market Analysis",
            confidence=0.8,
            sources=[{"url": "AI-generated", "credibility": 0.75}],
            abstraction_level="expert"
        )
        
        return insights
    
    async def update_knowledge_relationships(self):
        console.print("🔗 Updating knowledge graph relationships")
        concepts = await self.knowledge_graph.get_all_concepts()
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                relationship = await self.llm_relationship.send_input_async(
                    f"Determine the relationship between these two financial concepts:\n\n"
                    f"Concept 1: {concept1['name']} - {concept1['description']}\n"
                    f"Concept 2: {concept2['name']} - {concept2['description']}\n\n"
                    f"Provide a brief description of their relationship."
                )
                await self.knowledge_graph.add_relationship(concept1['name'], concept2['name'], relationship)

async def ai_powered_code_review_system():
    console.print(Panel("AI-Powered Code Review System 🛠️", title="Program 1"))

    # Initialize LLM and CodebaseEmbeddings
    console.print("\033✨ Initializing LLM and CodebaseEmbeddings... ✨\033")
    llm = LLM(provider="google", model="gemini-1.5-flash", config=LLMConfig(temperature=0.7, max_tokens=2000))
    embeddings = Embeddings(provider="openai", model="text-embedding-3-small")
    codebase_embeddings = CodebaseEmbeddings(embeddings)
    
    # Simulate loading a codebase
    codebase_path = "./llmcore"
    console.print(f"\033🔍 Building embeddings for codebase at {codebase_path} 🚀\033")
    await codebase_embeddings.build_embeddings(codebase_path)
    
    # Define review criteria
    review_criteria = [
        "Code quality and readability",
        "Performance optimizations",
        "Security vulnerabilities",
        "Adherence to best practices",
        "Documentation completeness"
    ]
    console.print("\033✅ Review criteria defined\033")
    
    # Create a prompt template for code review
    review_template = PromptTemplate(
        "Perform a code review for the following code snippet:\n\n"
        "```{{language}}\n{{code}}\n```\n\n"
        "Here are some relevant code snippets:\n\n"
        "```{{relevant_code}}\n```\n\n"
        "Consider the following criteria:\n{{criteria}}\n\n"
        "Provide a detailed review with specific suggestions for improvement.",
        required_params={"language": str, "code": str, "criteria": str, "relevant_code": str},
        output_json_structure={
            "overall_rating": int,
            "comments": List[Dict[str, Any]],
            "suggestions": List[str]
        }
    )
    
    # Simulate a code review process
    async def review_code_snippet(snippet, language):
        console.print(f"\033🔍 Reviewing code snippet in {language}\033")
        relevant_chunks = await codebase_embeddings.get_relevant_snippets(query=snippet, top_k=3, snippet_type="function")
        context = "\n".join([chunk.content for chunk in relevant_chunks])
        
        prompt = review_template.create_prompt(
            language=language,
            code=snippet,
            criteria="\n".join(review_criteria),
            relevant_code=context
        )
        
        review_result = await llm.send_input_async(prompt, parse_json=True)
        console.print("\033✅ Code review completed\033")
        return review_result
    
    # Example usage
    code_snippet = """
    # Example of interesting non-generic math-based code that shows
    # some extremely profound plots and analysis
    import matplotlib.pyplot as plt

    def fractal_plot():
        # Generate a fractal plot
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-2, 2, 1000)
        x, y = np.meshgrid(x, y)
        z = np.sqrt(x**2 + y**2)
        plt.imshow(z, cmap='hot', interpolation='bilinear')
        plt.colorbar()
        plt.show()

    if __name__ == "__main__":
        fractal_plot()
    """
    
    review = await review_code_snippet(code_snippet, "python")
    console.print(f"\033📝 Code Review Result: {review}\033")

async def ai_assisted_customer_support():
    console.print(Panel("AI-Assisted Customer Support System 🤖", title="Program 2"))
    
    llm = LLM(provider="anthropic", model="claude-3-5-sonnet-20240620", config=LLMConfig(temperature=0.7, max_tokens=2000))
    knowledge_graph = KnowledgeGraph()
    
    # Initialize knowledge graph with more detailed product information
    console.print("🧠 Initializing knowledge graph with product information")
    concept_ids = {}

    # Add concepts with initial relationships
    concept_ids["SmartHome Hub"] = await knowledge_graph.add_concept(
        name="SmartHome Hub",
        description="Central device for controlling smart home devices. Features include: "
                    "Wi-Fi and Bluetooth connectivity, voice control integration, "
                    "compatibility with major smart home protocols.",
        category=ConceptCategory.TECHNOLOGY,
        sources=[{"url": "https://example.com/smarthome-hub", "credibility": 0.95}],
        tags=["smart home", "hub", "central device"],
        significance=0.9,
        initial_relationships=[]  # No initial relationships
    )

    concept_ids["Mobile App"] = await knowledge_graph.add_concept(
        name="Mobile App",
        description="Companion app for controlling SmartHome Hub. Available on iOS and Android. "
                    "Features include: device management, automation creation, energy monitoring.",
        category=ConceptCategory.TECHNOLOGY,
        sources=[{"url": "https://example.com/mobile-app", "credibility": 0.9}],
        tags=["app", "mobile", "control"],
        significance=0.85,
        initial_relationships=[
            {
                'target_id': concept_ids["SmartHome Hub"],
                'rel_type': "controls",
                'strength': 0.9
            }
        ]
    )
    
    concept_ids["Hub-App Connection"] = await knowledge_graph.add_concept(
        name="Hub-App Connection",
        description="Process of linking the SmartHome Hub to the mobile app. Requires both devices "
                    "to be on the same Wi-Fi network. Uses QR code scanning for secure pairing.",
        category=ConceptCategory.TECHNOLOGY,
        sources=[{"url": "https://example.com/hub-app-connection", "credibility": 0.85}],
        tags=["connection", "pairing", "setup"],
        significance=0.8,
        initial_relationships=[
            {
                'target_id': concept_ids["SmartHome Hub"],
                'rel_type': "enables",
                'strength': 0.8
            },
            {
                'target_id': concept_ids["Mobile App"],
                'rel_type': "facilitates",
                'strength': 0.7
            }
        ]
    )
    
    # Display a summary of the updated knowledge graph
    console.print("\n📚 Updated Knowledge Graph Summary:")
    concepts = knowledge_graph.get_all_concepts()
    kg_table = Table(title="Knowledge Graph Concepts")
    kg_table.add_column("Concept", style="cyan")
    kg_table.add_column("Description", style="magenta")
    kg_table.add_column("Category", style="magenta")
    kg_table.add_column("Tags", style="green")
    kg_table.add_column("Significance", style="green")
    for concept in concepts:
        tags = ", ".join(concept.tags)
        kg_table.add_row(concept.name, concept.description, concept.category.value, tags, f"{concept.significance:.2f}")
    console.print(kg_table)
    
    # Display a summary of the relationships in the knowledge graph
    console.print("\n🔗 Knowledge Graph Relationships:")
    relationships = knowledge_graph.get_all_relationships()
    rel_table = Table(title="Knowledge Graph Relationships")
    rel_table.add_column("Source", style="cyan")
    rel_table.add_column("Target", style="magenta")
    rel_table.add_column("Type", style="green")
    rel_table.add_column("Strength", style="green")
    for rel in relationships:
        rel_table.add_row(rel["source"], rel["target"], rel["type"], f"{rel['strength']:.2f}")
    console.print(rel_table)
    
    # Create a more comprehensive prompt template for customer support
    support_template = PromptTemplate(
        "You are an AI customer support agent for a smart home company. "
        "Use the following product information to answer the customer's question:\n\n"
        "{{product_info}}\n\n"
        "Customer question: {{question}}\n\n"
        "Provide a helpful, accurate, and detailed response. Include step-by-step instructions if applicable. "
        "Also, suggest related topics or features that might be of interest to the customer.",
        required_params={"product_info": str, "question": str},
        output_json_structure={
            "response": str,
            "confidence": float,
            "follow_up_questions": List[str],
            "related_topics": List[str]
        }
    )
    
    async def handle_customer_query(question: str):
        # Query the knowledge graph for relevant concepts
        relevant_concepts = await knowledge_graph.query_graph(question)
        
        # Get related concepts to provide more context
        related_concepts = []
        for concept in relevant_concepts:
            related = await knowledge_graph.get_related_concepts(concept['id'])
            related_concepts.extend(related)
        
        # Combine all relevant information
        product_info = "\n\n".join([
            f"Concept: {concept['name']}\nDescription: {concept['description']}\nSignificance: {concept['significance']}"
            for concept in relevant_concepts + related_concepts
        ])
        
        prompt = support_template.create_prompt(
            product_info=product_info,
            question=question
        )
        
        response = await llm.send_input_async(prompt, parse_json=True)
        return response
    
    # Example usage with multiple questions
    questions = [
        "How do I connect my SmartHome Hub to my mobile app?",
        "Can I control my smart lights using voice commands?",
        "What should I do if my hub is not detecting new devices?"
    ]
    
    for question in questions:
        console.print(f"\n🤔 Customer Question: {question}")
        answer = await handle_customer_query(question)
        
        # Display the response in a structured format
        console.print(Panel(answer["response"], title="📩 Customer Support Response", expand=False))
        console.print(f"Confidence: {answer['confidence']:.2f}")
        
        if answer["follow_up_questions"]:
            console.print("Follow-up Questions:")
            for q in answer["follow_up_questions"]:
                console.print(f"  • {q}")
        
        if answer["related_topics"]:
            console.print("Related Topics:")
            for topic in answer["related_topics"]:
                console.print(f"  • {topic}")
        
        # Update knowledge graph based on the interaction
        category_list_string = "\n".join([category.value for category in ConceptCategory])
        knowledge_analysis_template = PromptTemplate("""Analyze the following customer support interaction for new, valuable knowledge to add to our knowledge graph:
Question: {{question}}
Response: {{response}}

If there's significant new information, provide the following in JSON format:
1. A concise name for the concept (max 50 characters)
2. A brief description of the new knowledge
3. The category it belongs to from the following options:
{{category_list}}
4. A significance score (0.0 to 1.0) based on the importance of the information
5. Relevant tags for the concept

IMPORTANT: If there's no significant new information, return empty strings, 0.0 for significance, and an empty list for tags.""",
            required_params={"question": str, "response": str},
            output_json_structure={
                "name": str,
                "description": str,
                "category": str,
                "significance": float,
                "tags": List[str]
            }
        )
    
        knowledge_analysis_prompt = knowledge_analysis_template.create_prompt(
            question=question,
            response=answer["response"],
            category_list=category_list_string
        )
    
        # Use a faster, smaller model for this analysis to keep the process quick
        knowledge_analysis_llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.4, max_tokens=1024))
        knowledge_analysis = await knowledge_analysis_llm.send_input_async(knowledge_analysis_prompt, parse_json=True)
    
        if knowledge_analysis and knowledge_analysis.get("name"):
            # Add the new concept
            new_concept_id = await knowledge_graph.add_concept(
                name=knowledge_analysis["name"],
                description=knowledge_analysis["description"],
                category=knowledge_analysis["category"],
                sources=[{"url": "AI-generated from customer interaction", "credibility": 0.8}],
                tags=knowledge_analysis.get("tags", []),
                significance=knowledge_analysis.get("significance", 1.0)
            )
            console.print(f"[bold green]New knowledge added to graph: {knowledge_analysis['name']}[/bold green]")
            
            # Optionally, establish relationships between the new concept and existing concepts
            # This can be based on predefined rules or additional LLM prompts
            # For simplicity, we'll create a generic 'related_to' relationship with a random existing concept
            # Here, we choose 'SmartHome Hub' as an example
            await knowledge_graph.add_relationship(
                source_id=new_concept_id,
                target_id=concept_ids["SmartHome Hub"],
                rel_type="related_to",
                strength=0.5
            )
            console.print(f"[bold blue]Established 'related_to' relationship between '{knowledge_analysis['name']}' and 'SmartHome Hub'.[/bold blue]")
        else:
            console.print("[yellow]No new significant knowledge to add to the graph.[/yellow]")
    
    # Display a summary of the updated knowledge graph
    console.print("\n📚 Updated Knowledge Graph Summary:")
    concepts = knowledge_graph.get_all_concepts()
    kg_table = Table(title="Knowledge Graph Concepts")
    kg_table.add_column("Concept", style="cyan")
    kg_table.add_column("Description", style="magenta")
    kg_table.add_column("Category", style="magenta")
    kg_table.add_column("Tags", style="green")
    kg_table.add_column("Significance", style="green")
    for concept in concepts:
        tags = ", ".join(concept.tags)
        kg_table.add_row(concept.name, concept.description, concept.category.value, tags, f"{concept.significance:.2f}")
    console.print(kg_table)

    # Display a summary of the relationships in the knowledge graph
    console.print("\n🔗 Knowledge Graph Relationships:")
    relationships = knowledge_graph.get_all_relationships()
    rel_table = Table(title="Knowledge Graph Relationships")
    rel_table.add_column("Source", style="cyan")
    rel_table.add_column("Target", style="magenta")
    rel_table.add_column("Type", style="green")
    rel_table.add_column("Strength", style="green")
    for rel in relationships:
        rel_table.add_row(rel["source"], rel["target"], rel["type"], f"{rel['strength']:.2f}")
    console.print(rel_table)

async def ai_driven_content_marketing_platform():
    console.print(Panel("AI-Driven Content Marketing Platform 📰", title="Program 3"))
    
    llm = LLM(provider="google", model="gemini-1.5-flash", config=LLMConfig(temperature=0.8, max_tokens=3000))
    
    # Create prompt templates for different content types
    blog_post_template = PromptTemplate(
        "Create a blog post outline on the topic: {{topic}}\n"
        "Target audience: {{audience}}\n"
        "Desired word count: {{word_count}}\n"
        "Key points to cover:\n{{key_points}}\n"
        "Tone: {{tone}}\n\n"
        "Provide an outline with main sections and brief descriptions.",
        required_params={"topic": str, "audience": str, "word_count": int, "key_points": str, "tone": str},
        output_json_structure={
            "title": str,
            "introduction": str,
            "main_sections": List[Dict[str, str]],
            "conclusion": str,
            "estimated_word_count": int
        }
    )
    
    social_media_template = PromptTemplate(
        "Create a series of social media posts for the following campaign:\n"
        "Campaign theme: {{theme}}\n"
        "Target platform: {{platform}}\n"
        "Number of posts: {{num_posts}}\n"
        "Call-to-action: {{cta}}\n\n"
        "Provide engaging and platform-appropriate content for each post.",
        required_params={"theme": str, "platform": str, "num_posts": int, "cta": str},
        output_json_structure={
            "posts": List[Dict[str, str]]
        }
    )
    
    email_newsletter_template = PromptTemplate(
        "Design an email newsletter on the topic: {{topic}}\n"
        "Target audience: {{audience}}\n"
        "Key announcements:\n{{announcements}}\n"
        "Promotional offer: {{offer}}\n\n"
        "Create a compelling email structure with subject line and main content sections.",
        required_params={"topic": str, "audience": str, "announcements": str, "offer": str},
        output_json_structure={
            "subject_line": str,
            "preview_text": str,
            "sections": List[Dict[str, str]],
            "cta_button": Dict[str, str]
        }
    )
    
    # Create LLMChain for content generation workflow
    content_chain = (
        LLMChainBuilder(llm)
        .add_step(
            template=blog_post_template.template,
            output_key="blog_post_outline",
            required_params=blog_post_template.required_params,
            output_json_structure=blog_post_template.output_json_structure
        )
        .add_step(
            template=social_media_template.template,
            output_key="social_media_campaign",
            required_params=social_media_template.required_params,
            output_json_structure=social_media_template.output_json_structure
        )
        .add_step(
            template=email_newsletter_template.template,
            output_key="email_newsletter",
            required_params=email_newsletter_template.required_params,
            output_json_structure=email_newsletter_template.output_json_structure
        )
        .build()
    )
    
    # Example usage
    content_brief = {
        "topic": "The Future of Artificial Intelligence in Marketing",
        "audience": "Marketing professionals and business owners",
        "word_count": 1500,
        "key_points": "AI-driven personalization, Predictive analytics, Chatbots and virtual assistants, Ethical considerations",
        "tone": "Informative and forward-thinking",
        "theme": "AI Revolution in Marketing",
        "platform": "LinkedIn",
        "num_posts": 5,
        "cta": "Download our AI Marketing Guide",
        "announcements": "Upcoming AI in Marketing webinar, New AI-powered marketing tool launch",
        "offer": "25% off AI Marketing Strategy consultation"
    }
    
    console.print("🔍 Generating content marketing campaign...")
    result = await content_chain.execute_async(content_brief)
    console.print("✅ Content marketing campaign generated:")
    console.print(f"📰 Blog Post Outline: {result['blog_post_outline']}")
    console.print(f"📰 Social Media Campaign: {result['social_media_campaign']}")
    console.print(f"📰 Email Newsletter: {result['email_newsletter']}")

async def ai_powered_financial_analysis():
    console.print(Panel("AI-Powered Financial Analysis System 📈", title="Program 4"))
    
    financial_system = AIFinancialAnalysisSystem()
    
    # API keys (replace with your actual API keys or use environment variables)
    alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    newsapi_key = os.environ.get("NEWSAPI_KEY")
    
    if not alpha_vantage_api_key or not newsapi_key:
        console.print("[bold red]Error: API keys not found. Please set ALPHA_VANTAGE_API_KEY and NEWSAPI_KEY environment variables.[/bold red]")
        return
    
    # Fetch real stock data
    symbols = ["AAPL", "GOOGL", "MSFT"]
    stock_data = {}
    for symbol in symbols:
        data = await financial_system.fetch_stock_data(symbol, alpha_vantage_api_key)
        stock_data[symbol] = data
    
    console.print(f"Stock data: {stock_data}")
    
    # Fetch real news articles
    news_articles = await financial_system.fetch_news_articles("stock market", newsapi_key)
    
    # Analyze news
    news_impacts = await financial_system.analyze_news(news_articles)
    console.print("\n📰 News Impacts:")
    for i, impact in enumerate(news_impacts, 1):
        console.print(Panel(f"""
Article: {news_articles[i-1]['title']}
Market Impact: {impact['market_impact']}
Affected Sectors: {', '.join(impact['affected_sectors'])}
        """, title=f"News Impact {i}"))
    
    # Analyze stocks
    stock_analyses = {}
    for symbol, data in stock_data.items():
        # Filter relevant news for the current stock
        relevant_news = [article for article in news_articles if symbol.lower() in article['title'].lower() or symbol.lower() in article['description'].lower()]

        if len(relevant_news) == 0:
            console.print(f"No relevant news found for {symbol}. Attempting to search more articles...")
            stock_articles = await financial_system.fetch_news_articles(query = symbol, api_key=newsapi_key)
            relevant_news.extend(stock_articles)
        
        analysis = await financial_system.analyze_stock(symbol, str(data["daily_data"]), str(data["overview"]), relevant_news)
        stock_analyses[symbol] = analysis
        
        console.print(f"\n📊 Analysis for {symbol}:")
        table = Table(title=f"{symbol} Stock Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Recommendation", analysis['recommendation'])
        table.add_row("Risk Level", analysis['risk_level'])
        table.add_row("Analysis", analysis['analysis'])
        
        # Add some key metrics from the company overview
        overview = data["overview"]
        table.add_row("Market Cap", overview.get("MarketCapitalization", "N/A"))
        table.add_row("P/E Ratio", overview.get("PERatio", "N/A"))
        table.add_row("Dividend Yield", overview.get("DividendYield", "N/A"))
        table.add_row("52 Week High", overview.get("52WeekHigh", "N/A"))
        table.add_row("52 Week Low", overview.get("52WeekLow", "N/A"))
        
        console.print(table)
    
    # Generate market insights
    last_trading_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    market_data = f"S&P 500 closed at {stock_data['AAPL']['daily_data']['Time Series (Daily)'][last_trading_day]['4. close']} on {last_trading_day}"
    insights = await financial_system.generate_market_insights(market_data, news_impacts)
    console.print("\n💡 Market Insights:")
    insight_table = Table(title="Market Insights and Recommendations")
    insight_table.add_column("Recommendation", style="cyan")
    for recommendation in insights["recommendations"]:
        insight_table.add_row(recommendation)
    console.print(insight_table)
    console.print(Panel(f"Risk Assessment: {insights['risk_assessment']}", title="Risk Assessment"))
    
    # Update knowledge graph relationships
    await financial_system.update_knowledge_relationships()
    console.print("\n🧠 Knowledge graph updated with new relationships")
    
    # Display a summary of the knowledge graph
    concepts = await financial_system.knowledge_graph.get_all_concepts()
    console.print("\n📚 Knowledge Graph Summary:")
    kg_table = Table(title="Knowledge Graph Concepts")
    kg_table.add_column("Concept", style="cyan")
    kg_table.add_column("Category", style="magenta")
    kg_table.add_column("Confidence", style="green")
    for concept in concepts:
        kg_table.add_row(concept['name'], concept['category'], str(concept['confidence']))
    console.print(kg_table)

async def stream_llm_response():
    console.print(Panel("Streaming LLM Response Example", title="Additional Example 1"))
    
    llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.7, max_tokens=1000))
    
    prompt_template = PromptTemplate(
        "Write a short story about {{topic}} in {{style}} style. Provide the story in chunks.",
        required_params={"topic": str, "style": str}
    )
    
    console.print("🖋️ Generating a short story...")
    async for chunk in llm.stream_input_async(prompt_template.create_prompt(topic="time travel", style="science fiction")):
        console.print(chunk, end="")
    console.print("\n✅ Story generation complete!")

async def code_similarity_search():
    console.print(Panel("Code Similarity Search Example", title="Additional Example 2"))
    
    embeddings = Embeddings(provider="openai", model="text-embedding-3-small")
    codebase_llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0, max_tokens=2056, top_p=1))
    codebase_embeddings = CodebaseEmbeddings(embeddings, codebase_llm)

    console.print("🔍 Building embeddings for codebase at ./llmcore")
    await codebase_embeddings.build_embeddings("./llmcore")
    
    query = "What can I use to encode information in a text and use with LLMs"
    
    # Check if there are any snippets before trying to get relevant ones
    if not codebase_embeddings.snippets:
        console.print("[yellow]Warning: No code snippets found. Make sure the codebase has been properly indexed.[/yellow]")
    else:
        similar_snippets = await codebase_embeddings.get_relevant_snippets(query, top_k=5)
        
        if not similar_snippets:
            console.print("[yellow]No relevant snippets found for the given query.[/yellow]")
        else:
            console.print(f"🔍 Top {len(similar_snippets)} code snippets similar to '{query}':")
            for i, snippet in enumerate(similar_snippets, 1):
                console.print(f"\n[bold]Snippet {i}:[/bold]")
                console.print(f"File: {snippet.file_path}")
                console.print(f"Function: {snippet.name}")
                console.print(f"Similarity Score: {snippet.relevance_score:.2f}")
                console.print("Code:")
                console.print(snippet.content)

async def context_aware_conversation():
    console.print(Panel("Memory Stress Test: Building a Fictional Character", title="Additional Example 3"))
    
    llm = LLM(provider="google", model="gemini-1.5-flash", config=LLMConfig(temperature=1.2, max_tokens=500, top_p=1))
    
    conversation_template = PromptTemplate(
        template="""User: {{question}}""",
        required_params={"question": str},
        output_json_structure={"response": str}
    )
    
    # Simulate a conversation that builds upon previous information
    questions = [
        "Let's create a fictional character named Alex. What's their profession?",
        "Given Alex's profession, what might be their biggest challenge at work?",
        "How does Alex's work challenge affect their personal life?",
        "What hobby might Alex take up to cope with their stress?",
        "How might Alex's hobby lead to an unexpected opportunity in their career?"
    ]
    
    for question in questions:
        prompt = conversation_template.create_prompt(question=question)
        response = await llm.send_input_with_memory(prompt=prompt, parse_json=True)
        
        console.print(f"[bold]User:[/bold] {question}")
        if isinstance(response, dict):  
            console.print(f"[bold]AI:[/bold] {response['response']}")
        else:
            console.print(f"[bold]AI:[/bold] {response}")
        
        console.print("---")
        
        # Simulate a delay to allow for memory processing
        await asyncio.sleep(1)
    
    # Test long-term memory retention with a question that requires recalling the entire conversation
    final_question = "Summarize Alex's journey from their initial career challenges to their current situation, mentioning how their hobby played a role."
    
    prompt = conversation_template.create_prompt(question=final_question)
    response = await llm.send_input_with_memory(prompt=prompt, parse_json=True)
    
    console.print(f"[bold]User (Final Question):[/bold] {final_question}")
    if isinstance(response, dict):
        console.print(f"[bold]AI:[/bold] {response['response']}")
    else:
        console.print(f"[bold]AI:[/bold] {response}")
    console.print("---")

async def multi_step_analysis():
    console.print(Panel("Multi-Step Analysis Example", title="Additional Example 4"))
    
    llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.8, max_tokens=2000))
    
    analysis_chain = (
        LLMChainBuilder(llm)
        .add_step(
            template="Summarize the following text in 2-3 sentences:\n\n{{text}}",
            output_key="summary",
            required_params={"text": str},
            output_json_structure={"summary": str, "paraphrased_summary": str}
        )
        .add_step(
            template="Identify the main themes in this summary:\n\n{{summary.summary}}\n\nParaphrased summary: {{summary.paraphrased_summary}}",
            output_key="themes",
            required_params={"summary": Dict[str, str]},
            output_json_structure={"themes": List[str]}
        )
        .add_step(
            template="Based on these themes, suggest 3 follow-up questions:\n\nThemes: {{themes}}",
            output_key="questions",
            required_params={"themes": Dict[str, Any]},
            output_json_structure={"questions": List[str]}
        )
        .build()
    )
    
    text = """
    Climate change is one of the most pressing issues of our time. It affects every aspect of our lives, 
    from the food we eat to the air we breathe. Scientists warn that without immediate action, we face 
    severe consequences including rising sea levels, more frequent natural disasters, and disruptions 
    to ecosystems worldwide. Governments and organizations are working to implement sustainable practices 
    and reduce carbon emissions, but many argue that more drastic measures are needed to address this 
    global crisis effectively.
    """
    
    console.print("🔍 Performing multi-step analysis...")
    result = await analysis_chain.execute_async({"text": text})
    
    console.print("\n[bold]Summary:[/bold]")
    console.print(result["summary"]["summary"])

    console.print("\n[bold]Paraphrased Summary:[/bold]")
    console.print(result["summary"]["paraphrased_summary"])
    
    console.print("\n[bold]Main Themes:[/bold]")
    for theme in result["themes"]["themes"]:
        console.print(f"- {theme}")
    
    console.print("\n[bold]Follow-up Questions:[/bold]")
    for question in result["questions"]["questions"]:
        console.print(f"- {question}")

async def main():
    programs = {
        "1": ("AI-Powered Code Review System", ai_powered_code_review_system),
        "2": ("AI-Assisted Customer Support", ai_assisted_customer_support),
        "3": ("AI-Driven Content Marketing Platform", ai_driven_content_marketing_platform),
        "4": ("AI-Powered Financial Analysis System", ai_powered_financial_analysis),
        "5": ("Streaming LLM Response", stream_llm_response),
        "6": ("Code Similarity Search", code_similarity_search),
        "7": ("Context-Aware Conversation", context_aware_conversation),
        "8": ("Multi-Step Analysis", multi_step_analysis),
    }

    while True:
        console.print(Panel("Select a program to run:", title="Menu"))
        for key, (name, _) in programs.items():
            console.print(f"[bold]{key}[/bold]: {name}")
        console.print("[bold]q[/bold]: Quit")

        choice = Prompt.ask("Enter your choice", choices=list(programs.keys()) + ["q"])

        if choice == "q":
            break

        program_name, program_func = programs[choice]
        console.print(f"\nRunning [bold]{program_name}[/bold]...")
        await program_func()
        console.print("\nProgram completed. Press Enter to continue...")
        input()

if __name__ == "__main__":
    asyncio.run(main())