"""
Service Discovery Module - Consul Integration
AquaWatch Project

Usage:
    from service_discovery import register_service, discover_service
    register_service('my-service', port=5000, health_path='/health')
"""

import os
import atexit
import signal
import consul

CONSUL_HOST = os.getenv('CONSUL_HOST', 'consul')
CONSUL_PORT = int(os.getenv('CONSUL_PORT', '8500'))

_consul_client = None
_registered_service_id = None


def get_consul():
    """Get or create Consul client."""
    global _consul_client
    if _consul_client is None:
        _consul_client = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
    return _consul_client


def register_service(service_name, port=0, health_path=None):
    """
    Register a service with Consul.
    
    Args:
        service_name: Name of the service
        port: Port the service listens on (0 if no HTTP endpoint)
        health_path: Health check endpoint path (e.g., '/health')
    
    Returns:
        Service ID
    """
    global _registered_service_id
    
    c = get_consul()
    service_id = f"{service_name}-{os.getpid()}"
    
    check = None
    if port > 0 and health_path:
        check = consul.Check.http(
            f'http://{service_name}:{port}{health_path}',
            interval='10s',
            timeout='5s',
            deregister='1m'
        )
    elif port > 0:
        check = consul.Check.tcp(
            f'{service_name}:{port}',
            interval='10s',
            timeout='5s'
        )
    
    try:
        c.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=service_name,
            port=port,
            check=check
        )
        _registered_service_id = service_id
        print(f'✅ Service "{service_name}" registered in Consul (ID: {service_id})')
        return service_id
    except Exception as e:
        print(f'❌ Failed to register service in Consul: {e}')
        raise


def discover_service(service_name):
    """
    Discover a service by name.
    
    Args:
        service_name: Name of the service to discover
    
    Returns:
        dict with 'host' and 'port' keys
    """
    c = get_consul()
    
    try:
        _, services = c.catalog.service(service_name)
        
        if not services:
            raise Exception(f'Service "{service_name}" not found in Consul')
        
        service = services[0]
        return {
            'host': service['ServiceAddress'] or service['Address'],
            'port': service['ServicePort']
        }
    except Exception as e:
        print(f'❌ Failed to discover service "{service_name}": {e}')
        raise


def discover_all_instances(service_name):
    """
    Get all instances of a service.
    
    Args:
        service_name: Name of the service
    
    Returns:
        List of dicts with 'host' and 'port' keys
    """
    c = get_consul()
    
    try:
        _, services = c.catalog.service(service_name)
        return [
            {
                'host': s['ServiceAddress'] or s['Address'],
                'port': s['ServicePort']
            }
            for s in services
        ]
    except Exception as e:
        print(f'❌ Failed to discover service instances: {e}')
        return []


def deregister_service():
    """Deregister the current service from Consul."""
    global _registered_service_id
    
    if _registered_service_id is None:
        return
    
    try:
        c = get_consul()
        c.agent.service.deregister(_registered_service_id)
        print(f'✅ Service deregistered from Consul (ID: {_registered_service_id})')
        _registered_service_id = None
    except Exception as e:
        print(f'❌ Failed to deregister service: {e}')


def is_consul_available():
    """Check if Consul is available."""
    try:
        c = get_consul()
        c.agent.members()
        return True
    except:
        return False


def wait_for_consul(max_retries=30, delay_seconds=2):
    """
    Wait for Consul to be available.
    
    Args:
        max_retries: Maximum number of retries
        delay_seconds: Delay between retries
    
    Returns:
        True if Consul is available, False otherwise
    """
    import time
    
    for i in range(max_retries):
        if is_consul_available():
            print('✅ Consul is available')
            return True
        print(f'⏳ Waiting for Consul... ({i + 1}/{max_retries})')
        time.sleep(delay_seconds)
    
    print('❌ Consul not available after retries')
    return False


# Graceful shutdown
def _cleanup_handler(signum=None, frame=None):
    deregister_service()
    if signum:
        exit(0)

atexit.register(deregister_service)
signal.signal(signal.SIGTERM, _cleanup_handler)
signal.signal(signal.SIGINT, _cleanup_handler)
