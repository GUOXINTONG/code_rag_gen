import React, { useState } from 'react';
import { ChevronRight, Brain, Code, Lightbulb, Network, Zap } from 'lucide-react';

const InterDisciplinaryQAGenerator = () => {
  const [inputCode, setInputCode] = useState('');
  const [step, setStep] = useState(0);
  const [results, setResults] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // 模拟的跨学科知识图谱
  const knowledgeGraph = {
    // 核心编程概念
    programming_concepts: {
      'sorting': {
        'properties': ['comparison', 'ordering', 'optimization'],
        'complexity': 'O(n log n)',
        'patterns': ['divide_conquer', 'recursive']
      },
      'binary_search': {
        'properties': ['logarithmic', 'divide_conquer', 'sorted_data'],
        'complexity': 'O(log n)',
        'patterns': ['elimination', 'bisection']
      },
      'dynamic_programming': {
        'properties': ['optimal_substructure', 'overlapping_subproblems'],
        'complexity': 'varies',
        'patterns': ['memoization', 'tabulation']
      }
    },
    
    // 跨域概念映射
    cross_domain_mappings: {
      'sorting': {
        'biology': ['natural_selection', 'evolutionary_pressure', 'fitness_ranking'],
        'economics': ['market_efficiency', 'price_discovery', 'resource_allocation'],
        'physics': ['entropy_reduction', 'energy_minimization', 'crystallization'],
        'psychology': ['cognitive_load', 'categorization', 'pattern_recognition'],
        'sociology': ['social_stratification', 'hierarchy_formation'],
        'game_theory': ['ranking_systems', 'tournament_structures']
      },
      'binary_search': {
        'biology': ['binary_fission', 'phylogenetic_trees', 'species_classification'],
        'economics': ['market_bisection', 'price_binary_search', 'supply_demand_equilibrium'],
        'physics': ['binary_phase_transitions', 'quantum_measurement', 'signal_processing'],
        'psychology': ['decision_trees', 'binary_choice_models'],
        'game_theory': ['perfect_information_games', 'minimax_trees'],
        'medicine': ['diagnostic_trees', 'symptom_elimination']
      },
      'dynamic_programming': {
        'biology': ['protein_folding', 'evolutionary_optimization', 'genetic_algorithms'],
        'economics': ['optimal_consumption', 'investment_strategies', 'resource_planning'],
        'physics': ['path_integral', 'variational_principles', 'phase_transitions'],
        'psychology': ['learning_curves', 'habit_formation', 'decision_optimization'],
        'game_theory': ['sequential_games', 'backward_induction'],
        'operations_research': ['supply_chain_optimization', 'scheduling']
      }
    },

    // 约束和应用场景
    constraints: {
      'performance': ['real_time', 'memory_limited', 'distributed'],
      'ethical': ['fairness', 'privacy', 'transparency', 'bias_mitigation'],
      'economic': ['cost_optimization', 'scalability', 'maintainability'],
      'physical': ['energy_efficiency', 'hardware_limitations', 'fault_tolerance'],
      'social': ['accessibility', 'user_experience', 'cultural_sensitivity']
    },

    // 应用领域
    application_domains: {
      'healthcare': ['medical_diagnosis', 'drug_discovery', 'patient_monitoring'],
      'finance': ['risk_assessment', 'fraud_detection', 'algorithmic_trading'],
      'gaming': ['AI_behavior', 'procedural_generation', 'player_matching'],
      'education': ['adaptive_learning', 'knowledge_tracing', 'personalization'],
      'environment': ['climate_modeling', 'resource_management', 'sustainability'],
      'security': ['cryptography', 'intrusion_detection', 'access_control']
    }
  };

  // 代码分析器
  const analyzeCode = (code) => {
    const concepts = [];
    const patterns = [];
    
    // 简单的模式匹配来识别概念
    if (code.includes('sort') || code.includes('Sort')) {
      concepts.push('sorting');
      patterns.push('comparison', 'ordering');
    }
    
    if (code.includes('binary') || code.includes('Binary') || 
        (code.includes('left') && code.includes('right')) ||
        code.includes('mid')) {
      concepts.push('binary_search');
      patterns.push('divide_conquer', 'elimination');
    }
    
    if (code.includes('dp') || code.includes('memo') || 
        code.includes('Dynamic') || code.includes('cache')) {
      concepts.push('dynamic_programming');
      patterns.push('memoization', 'optimization');
    }
    
    if (code.includes('for') || code.includes('while')) {
      patterns.push('iteration');
    }
    
    if (code.includes('def ') && code.includes('return')) {
      patterns.push('recursive');
    }

    return { concepts, patterns };
  };

  // 跨域概念选择器
  const selectCrossDomainConcepts = (primaryConcepts) => {
    const domains = ['biology', 'economics', 'physics', 'psychology', 'game_theory', 'medicine'];
    const selectedCombinations = [];
    
    primaryConcepts.forEach(concept => {
      if (knowledgeGraph.cross_domain_mappings[concept]) {
        domains.forEach(domain => {
          if (knowledgeGraph.cross_domain_mappings[concept][domain]) {
            const domainConcepts = knowledgeGraph.cross_domain_mappings[concept][domain];
            selectedCombinations.push({
              primary: concept,
              domain: domain,
              concepts: domainConcepts.slice(0, 2), // 取前两个概念
              constraint: Object.keys(knowledgeGraph.constraints)[Math.floor(Math.random() * Object.keys(knowledgeGraph.constraints).length)],
              application: Object.keys(knowledgeGraph.application_domains)[Math.floor(Math.random() * Object.keys(knowledgeGraph.application_domains).length)]
            });
          }
        });
      }
    });
    
    // 随机选择3个不同的组合
    const shuffled = selectedCombinations.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, 3);
  };

  // 问题生成器
  const generateQuestions = (combinations) => {
    const templates = {
      'biology': [
        "Design a {primary} algorithm inspired by {concept1} and {concept2}. Consider how {application} systems might implement this with {constraint} constraints.",
        "Implement a bio-inspired {primary} system that mimics {concept1} behavior. Apply this to {application} while ensuring {constraint}.",
        "Create a {primary} solution that models {concept1} processes. Optimize for {application} scenarios with {constraint} considerations."
      ],
      'economics': [
        "Develop a {primary} algorithm based on {concept1} and {concept2} principles. Apply economic modeling to {application} with {constraint} optimization.",
        "Design a market-driven {primary} system using {concept1} theory. Focus on {application} applications while maintaining {constraint}.",
        "Implement a {primary} solution inspired by {concept1} mechanisms. Consider {application} use cases with {constraint} trade-offs."
      ],
      'physics': [
        "Create a {primary} algorithm based on {concept1} and {concept2} principles from physics. Apply to {application} with {constraint} considerations.",
        "Design a physics-inspired {primary} system modeling {concept1} behavior. Optimize for {application} scenarios ensuring {constraint}.",
        "Implement a {primary} solution using {concept1} analogies. Focus on {application} applications with {constraint} requirements."
      ],
      'psychology': [
        "Develop a {primary} algorithm inspired by {concept1} and cognitive {concept2}. Apply to {application} systems with {constraint} design principles.",
        "Create a psychologically-motivated {primary} system based on {concept1} models. Optimize for {application} with {constraint} considerations.",
        "Design a {primary} solution that incorporates {concept1} principles. Focus on {application} use cases ensuring {constraint}."
      ],
      'game_theory': [
        "Implement a {primary} algorithm using {concept1} and {concept2} from game theory. Apply to {application} with {constraint} strategy optimization.",
        "Design a strategic {primary} system based on {concept1} principles. Focus on {application} scenarios with {constraint} considerations.",
        "Create a game-theoretic {primary} solution modeling {concept1} interactions. Optimize for {application} while ensuring {constraint}."
      ],
      'medicine': [
        "Develop a {primary} algorithm inspired by {concept1} and medical {concept2}. Apply to {application} systems with {constraint} requirements.",
        "Design a medically-inspired {primary} system based on {concept1} processes. Optimize for {application} with {constraint} considerations.",
        "Implement a {primary} solution using {concept1} analogies from medicine. Focus on {application} applications ensuring {constraint}."
      ]
    };

    return combinations.map((combo, index) => {
      const template = templates[combo.domain][index % templates[combo.domain].length];
      const question = template
        .replace('{primary}', combo.primary.replace('_', ' '))
        .replace('{concept1}', combo.concepts[0]?.replace('_', ' ') || 'optimization')
        .replace('{concept2}', combo.concepts[1]?.replace('_', ' ') || 'adaptation')
        .replace('{application}', combo.application.replace('_', ' '))
        .replace('{constraint}', combo.constraint.replace('_', ' '));
      
      return {
        ...combo,
        question: question,
        difficulty: ['Intermediate', 'Advanced', 'Expert'][index],
        tags: [combo.primary, combo.domain, combo.application, combo.constraint]
      };
    });
  };

  const handleGenerate = async () => {
    if (!inputCode.trim()) {
      alert('Please enter some code first!');
      return;
    }

    setIsGenerating(true);
    setStep(1);

    // 模拟处理步骤
    await new Promise(resolve => setTimeout(resolve, 1000));
    const analysis = analyzeCode(inputCode);
    
    setStep(2);
    await new Promise(resolve => setTimeout(resolve, 1000));
    const combinations = selectCrossDomainConcepts(analysis.concepts);
    
    setStep(3);
    await new Promise(resolve => setTimeout(resolve, 1000));
    const questions = generateQuestions(combinations);
    
    setResults({
      analysis,
      combinations,
      questions
    });
    
    setStep(4);
    setIsGenerating(false);
  };

  const resetDemo = () => {
    setStep(0);
    setResults(null);
    setInputCode('');
    setIsGenerating(false);
  };

  const sampleCode = `def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result`;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 min-h-screen">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-3">
          <Brain className="text-purple-600" />
          跨学科代码问题生成器
        </h1>
        <p className="text-gray-600 text-lg">基于扩展知识图谱的创新编程问题生成</p>
      </div>

      {step === 0 && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="mb-6">
            <label className="block text-lg font-semibold text-gray-700 mb-3">
              <Code className="inline mr-2" size={20} />
              输入你的代码:
            </label>
            <textarea
              value={inputCode}
              onChange={(e) => setInputCode(e.target.value)}
              placeholder="粘贴你的代码在这里..."
              className="w-full h-64 p-4 border-2 border-gray-200 rounded-lg font-mono text-sm resize-none focus:border-purple-400 focus:outline-none"
            />
          </div>
          
          <div className="flex gap-4">
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all duration-200 flex items-center justify-center gap-2"
            >
              <Zap size={20} />
              生成创新问题
            </button>
            
            <button
              onClick={() => setInputCode(sampleCode)}
              className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:border-gray-400 transition-colors"
            >
              使用示例代码
            </button>
          </div>
        </div>
      )}

      {(step > 0 && step < 4) && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center justify-center mb-8">
            <div className="flex items-center gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    i <= step ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-400'
                  }`}>
                    {i}
                  </div>
                  {i < 3 && <ChevronRight className="mx-2 text-gray-400" />}
                </div>
              ))}
            </div>
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center gap-3 text-lg font-semibold text-gray-700 mb-4">
              <div className="animate-spin">
                {step === 1 && <Code className="text-purple-600" />}
                {step === 2 && <Network className="text-blue-600" />}
                {step === 3 && <Lightbulb className="text-green-600" />}
              </div>
              {step === 1 && "分析代码结构和核心概念..."}
              {step === 2 && "映射跨学科知识图谱..."}
              {step === 3 && "生成创新问题..."}
            </div>
            <div className="w-64 bg-gray-200 rounded-full h-2 mx-auto">
              <div 
                className="bg-gradient-to-r from-purple-600 to-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${(step / 3) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {step === 4 && results && (
        <div className="space-y-8">
          {/* 分析结果 */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Code className="text-purple-600" />
              代码分析结果
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">识别的核心概念:</h3>
                <div className="flex flex-wrap gap-2">
                  {results.analysis.concepts.map((concept, i) => (
                    <span key={i} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
                      {concept.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">编程模式:</h3>
                <div className="flex flex-wrap gap-2">
                  {results.analysis.patterns.map((pattern, i) => (
                    <span key={i} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                      {pattern.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 生成的问题 */}
          <div className="grid gap-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Lightbulb className="text-green-600" />
              生成的跨学科编程问题
            </h2>
            
            {results.questions.map((q, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-xl font-bold text-gray-800">
                    问题 {index + 1}: {q.domain.charAt(0).toUpperCase() + q.domain.slice(1)} 视角
                  </h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    q.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                    q.difficulty === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {q.difficulty}
                  </span>
                </div>
                
                <p className="text-gray-700 text-lg leading-relaxed mb-4">
                  {q.question}
                </p>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  {q.tags.map((tag, i) => (
                    <span key={i} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                      #{tag.replace('_', ' ')}
                    </span>
                  ))}
                </div>
                
                <div className="text-sm text-gray-500">
                  <strong>跨域概念:</strong> {q.concepts.join(', ').replace(/_/g, ' ')} |
                  <strong> 应用领域:</strong> {q.application.replace('_', ' ')} |
                  <strong> 约束条件:</strong> {q.constraint.replace('_', ' ')}
                </div>
              </div>
            ))}
          </div>

          <div className="text-center">
            <button
              onClick={resetDemo}
              className="bg-gradient-to-r from-gray-600 to-gray-700 text-white py-3 px-8 rounded-lg font-semibold hover:from-gray-700 hover:to-gray-800 transition-all duration-200"
            >
              重新开始
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default InterDisciplinaryQAGenerator;