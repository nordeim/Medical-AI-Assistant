#!/usr/bin/env python3
"""
Model Serving CLI

Command-line interface for starting and managing model serving with adapter support.
Provides configuration management, health checks, monitoring, and graceful shutdown.
"""

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from model_serving import ModelServingManager, create_fastapi_app, ServingLauncher

console = Console()


class ServingCLI:
    """Command-line interface for model serving."""
    
    def __init__(self):
        self.manager: Optional[ModelServingManager] = None
        self.app: Optional[Any] = None
        self.server_process: Optional[uvicorn.Server] = None
        self.shutdown_event = asyncio.Event()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            console.print(f"\n[yellow]Received signal {signum}, initiating graceful shutdown...[/yellow]")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown routine."""
        console.print("[yellow]Shutting down server...[/yellow]")
        self.shutdown_event.set()
        
        if self.server_process:
            self.server_process.should_exit = True
        
        if self.manager and self.manager.adapter_manager:
            console.print("[yellow]Cleaning up adapters...[/yellow]")
            self.manager.adapter_manager.cleanup()
        
        console.print("[green]Shutdown complete[/green]")
        sys.exit(0)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            console.print(f"[green]✓ Loaded config from {config_path}[/green]")
            return config
        except FileNotFoundError:
            console.print(f"[red]✗ Config file not found: {config_path}[/red]")
            sys.exit(1)
        except yaml.YAMLError as e:
            console.print(f"[red]✗ Invalid YAML in config file: {e}[/red]")
            sys.exit(1)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration settings."""
        required_fields = ["model", "model.base_model_id"]
        
        # Check nested required fields
        if "model" not in config:
            console.print("[red]✗ Missing 'model' section in config[/red]")
            return False
        
        if "base_model_id" not in config["model"]:
            console.print("[red]✗ Missing 'base_model_id' in model config[/red]")
            return False
        
        # Validate optional sections
        if "adapters" in config:
            for adapter_id, adapter_config in config["adapters"].items():
                if "path" not in adapter_config:
                    console.print(f"[yellow]⚠ Warning: Adapter {adapter_id} missing 'path' field[/yellow]")
        
        if "server" in config:
            server_config = config["server"]
            if "port" in server_config and not (1 <= server_config["port"] <= 65535):
                console.print("[red]✗ Invalid port number (must be 1-65535)[/red]")
                return False
        
        if "resources" in config:
            resources_config = config["resources"]
            if "max_memory_mb" in resources_config:
                if not isinstance(resources_config["max_memory_mb"], (int, float)) or resources_config["max_memory_mb"] <= 0:
                    console.print("[red]✗ Invalid max_memory_mb value[/red]")
                    return False
        
        console.print("[green]✓ Configuration validation passed[/green]")
        return True
    
    async def initialize_manager(self, config: Dict[str, Any]) -> ModelServingManager:
        """Initialize the model serving manager."""
        console.print("[blue]Initializing model serving manager...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading base model...", total=None)
            
            try:
                manager = ModelServingManager(
                    base_model_id=config["model"]["base_model_id"],
                    adapter_configs=config.get("adapters", {}),
                    host=config.get("server", {}).get("host", "0.0.0.0"),
                    port=config.get("server", {}).get("port", 8000),
                    max_memory_mb=config.get("resources", {}).get("max_memory_mb", 8192)
                )
                
                await manager.initialize()
                
                progress.update(task, description="Base model loaded ✓")
                console.print("[green]✓ Model serving manager initialized[/green]")
                
                return manager
                
            except Exception as e:
                console.print(f"[red]✗ Failed to initialize manager: {e}[/red]")
                raise
    
    def display_startup_info(self, config: Dict[str, Any], manager: ModelServingManager):
        """Display startup information."""
        table = Table(title="Model Serving Startup Information")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Base Model", config["model"]["base_model_id"])
        table.add_row("Host", config.get("server", {}).get("host", "0.0.0.0"))
        table.add_row("Port", str(config.get("server", {}).get("port", 8000)))
        table.add_row("Adapters Loaded", str(len(manager.adapter_manager.list_adapters())))
        table.add_row("Active Adapter", manager.adapter_manager._active_adapter_id or "None")
        
        if "resources" in config:
            table.add_row("Max Memory (MB)", str(config["resources"].get("max_memory_mb", 8192)))
        
        console.print(table)
        
        # Display loaded adapters
        if manager.adapter_manager.list_adapters():
            adapter_table = Table(title="Loaded Adapters")
            adapter_table.add_column("Adapter ID", style="cyan")
            adapter_table.add_column("Version", style="yellow")
            adapter_table.add_column("Description", style="green")
            adapter_table.add_column("Size (MB)", style="blue")
            
            for adapter_id, metadata in manager.adapter_manager.list_adapters().items():
                adapter_table.add_row(
                    adapter_id,
                    metadata.version,
                    metadata.description or "N/A",
                    f"{metadata.file_size / (1024*1024):.1f}"
                )
            
            console.print(adapter_table)
        
        console.print("\n[bold green]Server is ready! Access it at:[/bold green]")
        console.print(f"  • Health Check: http://{config.get('server', {}).get('host', '0.0.0.0')}:{config.get('server', {}).get('port', 8000)}/health")
        console.print(f"  • API Documentation: http://{config.get('server', {}).get('host', '0.0.0.0')}:{config.get('server', {}).get('port', 8000)}/docs")
        console.print(f"  • Metrics: http://{config.get('server', {}).get('host', '0.0.0.0')}:{config.get('server', {}).get('port', 8000)}/metrics")
    
    async def run_server(self, config: Dict[str, Any]):
        """Run the server with graceful shutdown."""
        self.setup_signal_handlers()
        
        # Initialize manager
        self.manager = await self.initialize_manager(config)
        
        # Create FastAPI app
        self.app = create_fastapi_app(self.manager)
        
        # Display startup information
        self.display_startup_info(config, self.manager)
        
        # Configure server
        host = config.get("server", {}).get("host", "0.0.0.0")
        port = config.get("server", {}).get("port", 8000)
        workers = config.get("server", {}).get("workers", 1)
        log_level = config.get("server", {}).get("log_level", "info")
        
        # Create server config
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True
        )
        
        self.server_process = uvicorn.Server(config)
        
        console.print(f"\n[bold blue]Starting server on {host}:{port}[/bold blue]")
        
        try:
            await self.server_process.serve()
        except KeyboardInterrupt:
            console.print("\n[yellow]Received keyboard interrupt[/yellow]")
        finally:
            await self.shutdown()
    
    async def check_health(self, config: Dict[str, Any]):
        """Check server health."""
        import httpx
        
        host = config.get("server", {}).get("host", "localhost")
        port = config.get("server", {}).get("port", 8000)
        health_url = f"http://{host}:{port}/health"
        
        console.print(f"[blue]Checking health at {health_url}...[/blue]")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                response.raise_for_status()
                
                health_data = response.json()
                
                # Display health info
                table = Table(title="Server Health")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Status", health_data["status"])
                table.add_row("Version", health_data["version"])
                table.add_row("Adapters Loaded", str(health_data["adapters_loaded"]))
                table.add_row("Active Adapter", health_data["active_adapter"] or "None")
                table.add_row("Memory Usage (MB)", f"{health_data['memory_usage_mb']:.1f}")
                table.add_row("GPU Available", str(health_data["gpu_available"]))
                
                console.print(table)
                
                if health_data["status"] == "healthy":
                    console.print("[green]✓ Server is healthy[/green]")
                else:
                    console.print("[red]✗ Server is not healthy[/red]")
                    
        except httpx.RequestError as e:
            console.print(f"[red]✗ Failed to connect to server: {e}[/red]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗ Server returned error: {e}[/red]")
    
    async def list_adapters(self, config: Dict[str, Any]):
        """List loaded adapters."""
        import httpx
        
        host = config.get("server", {}).get("host", "localhost")
        port = config.get("server", {}).get("port", 8000)
        adapters_url = f"http://{host}:{port}/adapters"
        
        console.print(f"[blue]Fetching adapters from {adapters_url}...[/blue]")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(adapters_url)
                response.raise_for_status()
                
                adapters = response.json()
                
                if not adapters:
                    console.print("[yellow]No adapters loaded[/yellow]")
                    return
                
                table = Table(title="Loaded Adapters")
                table.add_column("Adapter ID", style="cyan")
                table.add_column("Version", style="yellow")
                table.add_column("Description", style="green")
                table.add_column("File Size (MB)", style="blue")
                table.add_column("Loaded At", style="magenta")
                table.add_column("Tags", style="dim")
                
                for adapter in adapters:
                    tags_str = ", ".join(adapter.get("tags", [])) if adapter.get("tags") else "None"
                    loaded_at = adapter["loaded_at"].split("T")[0] if "T" in adapter["loaded_at"] else adapter["loaded_at"]
                    
                    table.add_row(
                        adapter["adapter_id"],
                        adapter["version"],
                        adapter.get("description", "N/A"),
                        f"{adapter.get('file_size_mb', 0):.1f}",
                        loaded_at,
                        tags_str
                    )
                
                console.print(table)
                
        except httpx.RequestError as e:
            console.print(f"[red]✗ Failed to connect to server: {e}[/red]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗ Server returned error: {e}[/red]")
    
    async def switch_adapter(self, config: Dict[str, Any], adapter_id: str):
        """Switch to a different adapter."""
        import httpx
        
        host = config.get("server", {}).get("host", "localhost")
        port = config.get("server", {}).get("port", 8000)
        switch_url = f"http://{host}:{port}/adapters/{adapter_id}/switch"
        
        console.print(f"[blue]Switching to adapter: {adapter_id}[/blue]")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(switch_url)
                response.raise_for_status()
                
                result = response.json()
                console.print(f"[green]✓ {result['message']}[/green]")
                
        except httpx.RequestError as e:
            console.print(f"[red]✗ Failed to connect to server: {e}[/red]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗ Server returned error: {e}[/red]")
    
    async def hot_swap_adapter(self, config: Dict[str, Any], adapter_id: str, timeout: float = 30.0):
        """Hot-swap to a new adapter."""
        import httpx
        
        host = config.get("server", {}).get("host", "localhost")
        port = config.get("server", {}).get("port", 8000)
        hot_swap_url = f"http://{host}:{port}/adapters/{adapter_id}/hot-swap"
        
        console.print(f"[blue]Performing hot-swap to adapter: {adapter_id}[/blue]")
        console.print("[yellow]This may take a moment...[/yellow]")
        
        try:
            async with httpx.AsyncClient(timeout=timeout + 5.0) as client:
                response = await client.post(hot_swap_url, params={"timeout": timeout})
                response.raise_for_status()
                
                result = response.json()
                console.print(f"[green]✓ {result['message']}[/green]")
                
        except httpx.TimeoutException:
            console.print(f"[red]✗ Hot-swap timeout after {timeout} seconds[/red]")
        except httpx.RequestError as e:
            console.print(f"[red]✗ Failed to connect to server: {e}[/red]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗ Server returned error: {e}[/red]")
    
    async def get_metrics(self, config: Dict[str, Any]):
        """Get server performance metrics."""
        import httpx
        
        host = config.get("server", {}).get("host", "localhost")
        port = config.get("server", {}).get("port", 8000)
        metrics_url = f"http://{host}:{port}/metrics"
        
        console.print(f"[blue]Fetching metrics from {metrics_url}...[/blue]")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(metrics_url)
                response.raise_for_status()
                
                metrics = response.json()
                
                # Display metrics summary
                table = Table(title="Server Metrics")
                table.add_column("Metric Category", style="cyan")
                table.add_column("Details", style="green")
                
                # Adapter Manager Stats
                if "adapter_manager_stats" in metrics:
                    adapter_stats = metrics["adapter_manager_stats"]
                    table.add_row(
                        "Adapter Manager",
                        f"Adapters loaded: {adapter_stats.get('total_adapters_loaded', 0)}, "
                        f"Avg load time: {adapter_stats.get('avg_load_time', 0):.2f}s"
                    )
                
                # Serving Metrics
                if "serving_metrics" in metrics:
                    serving_metrics = metrics["serving_metrics"]
                    table.add_row(
                        "Request Errors",
                        str(serving_metrics.get("error_counts", {}))
                    )
                    table.add_row(
                        "Adapter Usage",
                        str(serving_metrics.get("adapter_usage", {}))
                    )
                
                # System Info
                if "system_info" in metrics:
                    sys_info = metrics["system_info"]
                    memory_stats = sys_info.get("memory_stats", {})
                    table.add_row(
                        "System Memory",
                        f"RSS: {memory_stats.get('rss_mb', 0):.1f}MB, "
                        f"Available: {memory_stats.get('available_mb', 0):.1f}MB"
                    )
                    table.add_row(
                        "GPU Status",
                        f"Available: {sys_info.get('gpu_available', False)}, "
                        f"Devices: {sys_info.get('device_count', 0)}"
                    )
                
                console.print(table)
                
        except httpx.RequestError as e:
            console.print(f"[red]✗ Failed to connect to server: {e}[/red]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗ Server returned error: {e}[/red]")
    
    def create_sample_config(self, output_path: str):
        """Create a sample configuration file."""
        sample_config = {
            "model": {
                "base_model_id": "microsoft/DialoGPT-medium"
            },
            "adapters": {
                "general_assistant": {
                    "path": "./adapters/general_assistant",
                    "description": "General purpose assistant adapter",
                    "tags": ["general", "assistant"]
                },
                "medical_specialist": {
                    "path": "./adapters/medical_specialist", 
                    "description": "Medical domain specialist adapter",
                    "tags": ["medical", "healthcare"]
                }
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info"
            },
            "resources": {
                "max_memory_mb": 8192
            },
            "security": {
                "allowed_hosts": ["*"],
                "cors_origins": ["*"]
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        
        console.print(f"[green]✓ Sample configuration created at {output_path}[/green]")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Model Serving CLI with Adapter Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with config
  python serve_model.py start --config serving_config.yaml
  
  # Check server health
  python serve_model.py health --config serving_config.yaml
  
  # List loaded adapters
  python serve_model.py list-adapters --config serving_config.yaml
  
  # Switch to specific adapter
  python serve_model.py switch --adapter general_assistant --config serving_config.yaml
  
  # Create sample config
  python serve_model.py create-config --output serving_config.yaml
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to serving configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the model serving server")
    start_parser.add_argument(
        "--host",
        type=str,
        help="Server host (overrides config)"
    )
    start_parser.add_argument(
        "--port", 
        type=int,
        help="Server port (overrides config)"
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of server workers"
    )
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check server health")
    
    # List adapters command
    list_parser = subparsers.add_parser("list-adapters", help="List loaded adapters")
    
    # Switch adapter command
    switch_parser = subparsers.add_parser("switch", help="Switch to different adapter")
    switch_parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter ID to switch to"
    )
    
    # Hot-swap command
    hot_swap_parser = subparsers.add_parser("hot-swap", help="Hot-swap to new adapter")
    hot_swap_parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter ID to hot-swap to"
    )
    hot_swap_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for hot-swap operation"
    )
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Get server metrics")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create sample configuration")
    config_parser.add_argument(
        "--output", "-o",
        type=str,
        default="serving_config.yaml",
        help="Output path for sample config"
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    cli = ServingCLI()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == "start":
            if not args.config:
                console.print("[red]✗ Config file required for start command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            if not cli.validate_config(config):
                sys.exit(1)
            
            # Override config with CLI args
            if args.host:
                config.setdefault("server", {})["host"] = args.host
            if args.port:
                config.setdefault("server", {})["port"] = args.port
            if args.workers:
                config.setdefault("server", {})["workers"] = args.workers
            
            await cli.run_server(config)
        
        elif args.command == "health":
            if not args.config:
                console.print("[red]✗ Config file required for health command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            await cli.check_health(config)
        
        elif args.command == "list-adapters":
            if not args.config:
                console.print("[red]✗ Config file required for list-adapters command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            await cli.list_adapters(config)
        
        elif args.command == "switch":
            if not args.config:
                console.print("[red]✗ Config file required for switch command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            await cli.switch_adapter(config, args.adapter)
        
        elif args.command == "hot-swap":
            if not args.config:
                console.print("[red]✗ Config file required for hot-swap command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            await cli.hot_swap_adapter(config, args.adapter, args.timeout)
        
        elif args.command == "metrics":
            if not args.config:
                console.print("[red]✗ Config file required for metrics command[/red]")
                sys.exit(1)
            
            config = cli.load_config(args.config)
            await cli.get_metrics(config)
        
        elif args.command == "create-config":
            cli.create_sample_config(args.output)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())