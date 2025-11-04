import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  AlertTriangle, 
  Clock, 
  Search,
  Filter,
  ArrowUpDown,
  Eye,
  CheckCircle
} from 'lucide-react';
import { NurseQueueItem, PARFilters, SortOption } from '@/types';
import { apiService } from '@/services/api';
import { formatDistanceToNow } from 'date-fns';

interface QueueViewProps {
  onPARSelect?: (parId: string) => void;
}

const QueueView: React.FC<QueueViewProps> = ({ onPARSelect }) => {
  const navigate = useNavigate();
  const [queueItems, setQueueItems] = useState<NurseQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState('');
  const [filters, setFilters] = useState<PARFilters>({});
  const [sort, setSort] = useState<SortOption>({ field: 'created_at', direction: 'desc' });
  const [stats, setStats] = useState({
    total: 0,
    urgent: 0,
    immediate: 0,
    redFlags: 0
  });

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    if (debouncedSearchQuery) {
      performSearch();
    } else {
      loadQueue();
    }
    const interval = setInterval(() => {
      if (!debouncedSearchQuery) {
        loadQueue();
      }
    }, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [debouncedSearchQuery, filters, sort]);

  const loadQueue = async () => {
    try {
      setLoading(true);
      const response = await apiService.getNurseQueue({
        limit: 100,
        offset: 0
      });
      
      if (response.success && response.data) {
        const data = response.data as any;
        setQueueItems(data.queue || data);
        setStats({
          total: data.total || data.length,
          urgent: data.urgent_count || 0,
          immediate: data.immediate_count || 0,
          redFlags: data.red_flag_count || 0
        });
      }
      setError(null);
    } catch (err) {
      setError('Failed to load queue');
      console.error('Queue load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const performSearch = async () => {
    try {
      setLoading(true);
      const response = await apiService.searchPARs(debouncedSearchQuery, {
        urgency: filters.urgency,
        risk_level: filters.risk_level,
        has_red_flags: filters.has_red_flags,
        date_from: filters.date_from,
        date_to: filters.date_to
      });
      
      if (response.success && response.data) {
        // Convert PAR data to queue items format
        const items = response.data.map((par: any) => ({
          id: par.id,
          session_id: par.session_id,
          patient_id: par.patient_id || 'unknown',
          patient_name: par.patient_name,
          chief_complaint: par.chief_complaint,
          risk_level: par.risk_level,
          urgency: par.urgency,
          has_red_flags: par.has_red_flags || false,
          created_at: par.created_at,
          wait_time_minutes: par.wait_time_minutes,
          status: par.status || 'pending'
        }));
        
        setQueueItems(items);
        setStats({
          total: items.length,
          urgent: items.filter((p: any) => p.urgency === 'urgent').length,
          immediate: items.filter((p: any) => p.urgency === 'immediate').length,
          redFlags: items.filter((p: any) => p.has_red_flags).length
        });
      }
      setError(null);
    } catch (err) {
      setError('Search failed');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-600 text-white border-red-700';
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getUrgencyIcon = (urgency: string) => {
    switch (urgency) {
      case 'immediate':
        return <AlertTriangle className="w-5 h-5 text-red-600" />;
      case 'urgent':
        return <Clock className="w-5 h-5 text-orange-600" />;
      default:
        return <Clock className="w-5 h-5 text-gray-600" />;
    }
  };

  const handleViewPAR = (parId: string) => {
    if (onPARSelect) {
      onPARSelect(parId);
    } else {
      navigate(`/nurse/par/${parId}`);
    }
  };

  const filteredAndSortedItems = queueItems
    .sort((a, b) => {
      const multiplier = sort.direction === 'asc' ? 1 : -1;
      
      switch (sort.field) {
        case 'urgency':
          const urgencyOrder = { immediate: 3, urgent: 2, routine: 1 };
          return (urgencyOrder[a.urgency] - urgencyOrder[b.urgency]) * multiplier;
        case 'risk_level':
          const riskOrder = { critical: 4, high: 3, medium: 2, low: 1 };
          return (riskOrder[a.risk_level] - riskOrder[b.risk_level]) * multiplier;
        case 'wait_time':
          return ((a.wait_time_minutes || 0) - (b.wait_time_minutes || 0)) * multiplier;
        default:
          return (new Date(a.created_at).getTime() - new Date(b.created_at).getTime()) * multiplier;
      }
    });

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Pending Reviews</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Immediate</p>
              <p className="text-2xl font-bold text-red-600">{stats.immediate}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Urgent</p>
              <p className="text-2xl font-bold text-orange-600">{stats.urgent}</p>
            </div>
            <Clock className="w-8 h-8 text-orange-600" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Red Flags</p>
              <p className="text-2xl font-bold text-red-600">{stats.redFlags}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
        </Card>
      </div>

      {/* Search and Filter */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search by complaint or patient name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Button variant="outline" onClick={() => {}}>
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </Button>
          <Button variant="outline" onClick={() => {}}>
            <ArrowUpDown className="w-4 h-4 mr-2" />
            Sort
          </Button>
        </div>
      </Card>

      {/* Queue List */}
      <Card>
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold text-gray-900">
            Patient Queue ({filteredAndSortedItems.length})
          </h2>
        </div>

        <div className="divide-y">
          {loading && (
            <div className="p-8 text-center text-gray-500">Loading queue...</div>
          )}

          {error && (
            <div className="p-8 text-center text-red-600">{error}</div>
          )}

          {!loading && !error && filteredAndSortedItems.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-600" />
              <p className="font-medium">No pending reviews</p>
              <p className="text-sm">All patients have been assessed</p>
            </div>
          )}

          {!loading && !error && filteredAndSortedItems.map((item) => (
            <div 
              key={item.id} 
              className="p-6 hover:bg-gray-50 transition-colors cursor-pointer"
              onClick={() => handleViewPAR(item.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    {getUrgencyIcon(item.urgency)}
                    <Badge className={getRiskLevelColor(item.risk_level)}>
                      {item.risk_level.toUpperCase()}
                    </Badge>
                    {item.has_red_flags && (
                      <Badge className="bg-red-100 text-red-800 border-red-200">
                        RED FLAGS
                      </Badge>
                    )}
                    <span className="text-sm text-gray-500">
                      {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                    </span>
                  </div>
                  
                  <h3 className="font-semibold text-gray-900 mb-2">
                    {item.patient_name || `Patient ${item.patient_id.slice(0, 8)}`}
                  </h3>
                  
                  <p className="text-gray-700 mb-2">{item.chief_complaint}</p>
                  
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>Urgency: {item.urgency}</span>
                    <span>•</span>
                    <span>Session: {item.session_id.slice(0, 8)}</span>
                    {item.wait_time_minutes && (
                      <>
                        <span>•</span>
                        <span>Wait: {item.wait_time_minutes} min</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  <Button variant="outline" size="sm">
                    <Eye className="w-4 h-4 mr-2" />
                    Review
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default QueueView;
